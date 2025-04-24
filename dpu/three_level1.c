#include <defs.h>
#include <mram.h>
#include <alloc.h>
#include <barrier.h>
#include <stdint.h>
#include <stdio.h>
#include <mutex.h>
#include <mutex_pool.h>
#include <built_ins.h>
#include <string.h>
#include <stdlib.h>

#include "pimindex.h"
#include "hash.h"

MUTEX_POOL_INIT(partitioning_mtx, 8);
BARRIER_INIT(index_barrier, NR_TASKLETS);

__host struct pimindex_dpu_args_t dpu_args;


/* MRAM pointers */
char* index_file_head;
char* index_file_head1;
char* index_file_head2;
char** chunks_offset;
IndexFile** index_file;
IndexFile** index_file1;
IndexFile** index_file2;

/* WRAM pointers */
void *wr_pool;
void **wr_buffer;
void **wr_buffer2;
void **wr_buffer3;
uint32_t** wr_chunk_bitmap;
#ifdef PINNED_CHUNK_HEADERS
ChunkHeader** wr_chunk_headers;
ChunkHeader** wr_chunk_headers1;
ChunkHeader** wr_chunk_headers2;
#endif

uint8_t* global_depth;
uint32_t* num_chunk_hdrs;
uint32_t* next_free_chunk_id;
uint32_t* index_file_lock; /* [63:32] unused, [31:] ptr, [30:] lock bit, [29:0] version */

uint32_t* wr_hist;
PIMKey_t* wr_part_buffer;
uint8_t* wr_part_counter;

uint32_t prefix;
const uint32_t index_file_lock_mask = (1u << 30);
const uint32_t mram_heap = (uint32_t)DPU_MRAM_HEAP_POINTER;


#ifdef HASH32
static inline uint32_t chunk_hdr_pos(HashValue_t hash, uint8_t mask_bits) {
    return (hash << 18) >> (32 - mask_bits);
}

static inline uint32_t bucket_id(HashValue_t hash) {
    return (hash << 8) >> 22;
}

static inline uint8_t fingerprint(HashValue_t hash) {
    return hash >> 24;
}
#else
static inline uint32_t chunk_hdr_pos(HashValue_t hash, uint8_t mask_bits) {
    return (hash << 32) >> (64 - mask_bits);
}

static inline uint32_t bucket_id(HashValue_t hash) {
    return (hash << 8) >> 54;
}

static inline uint8_t fingerprint(HashValue_t hash) {
    return hash >> 56;
}
#endif

static inline char* align_bucket_size(char* ptr) {
    /* return ptr += ((BUCKET_SIZE - ((uint32_t)ptr % BUCKET_SIZE)) % BUCKET_SIZE); */
    uint32_t rem = (uint32_t)ptr % BUCKET_SIZE;
    return (rem == 0) ? (ptr) : (ptr + BUCKET_SIZE - rem);
}

static inline void* chunk_ptr(uint32_t cid, uint32_t tasklet_id) {
    return chunks_offset[tasklet_id] + cid * CHUNK_SIZE;
}

uint32_t alloc_chunk(uint32_t tasklet_id) {
    uint32_t old_free_cid = next_free_chunk_id[tasklet_id];
    uint32_t byte_pos = old_free_cid / 32u;
    uint32_t old_val, new_val, bit_pos;
    byte_pos--;

    do {
        byte_pos++;
        old_val = wr_chunk_bitmap[tasklet_id][byte_pos];
    }
    while (old_val == 0);

    __builtin_clz_rr(bit_pos, old_val);
    new_val = old_val & ~(1u << (31 - bit_pos));
    wr_chunk_bitmap[tasklet_id][byte_pos] = new_val;

    uint32_t free_slot = byte_pos * 32 + bit_pos;
    next_free_chunk_id[tasklet_id] = free_slot + 1;

    return free_slot;
}

void initialize_index(uint32_t tasklet_id) {

    if (tasklet_id == 0) {
        printf("Tasklet: %u\n", tasklet_id);
        mem_reset();

        wr_buffer = (void**) mem_alloc(NR_TASKLETS * sizeof(void*));
        wr_chunk_bitmap = (uint32_t**) mem_alloc(NR_TASKLETS * sizeof(uint32_t*));
        num_chunk_hdrs = (uint32_t*) mem_alloc(NR_TASKLETS * sizeof(uint32_t));
        next_free_chunk_id = (uint32_t*) mem_alloc(NR_TASKLETS * sizeof(uint32_t));
        global_depth = (uint8_t*) mem_alloc(NR_TASKLETS * sizeof(uint8_t));
        index_file_lock = (uint32_t*) mem_alloc(NR_TASKLETS * sizeof(uint32_t));

        chunks_offset = (char**) mem_alloc(NR_TASKLETS * sizeof(char*));
        index_file = (IndexFile**) mem_alloc(NR_TASKLETS * sizeof(IndexFile*));
        index_file1 = (IndexFile**) mem_alloc(NR_TASKLETS * sizeof(IndexFile*));
        index_file2 = (IndexFile**) mem_alloc(NR_TASKLETS * sizeof(IndexFile*));
#ifdef PINNED_CHUNK_HEADERS
        wr_chunk_headers =
                (ChunkHeader**) mem_alloc(NR_TASKLETS * sizeof(ChunkHeader*));
        wr_chunk_headers1 =
                (ChunkHeader**) mem_alloc(NR_TASKLETS * sizeof(ChunkHeader*));
        wr_chunk_headers2 =
                (ChunkHeader**) mem_alloc(NR_TASKLETS * sizeof(ChunkHeader*));
#endif
    }
    barrier_wait(&index_barrier);

    wr_buffer[tasklet_id] = mem_alloc(WRAM_BUFFER_SIZE);
    wr_chunk_bitmap[tasklet_id] =
            (uint32_t*) mem_alloc(BITMAP_LEN_PER_TASKLET * sizeof(uint32_t));
    for (uint32_t i = 0; i < BITMAP_LEN_PER_TASKLET; i++) {
        wr_chunk_bitmap[tasklet_id][i] = (uint32_t)(-1); /* set all bits */
    }
    num_chunk_hdrs[tasklet_id] = (1u << INIT_GLOBAL_DEPTH);
    global_depth[tasklet_id] = INIT_GLOBAL_DEPTH;
    next_free_chunk_id[tasklet_id] = 0;
    index_file_lock[tasklet_id] = 0;

    char* ptr = (char*) (mram_heap + MRAM_INDEX_OFFSET +
                            tasklet_id * MAX_MRAM_SIZE_PER_TASKLET);

    ptr = align_bucket_size(ptr);
    index_file1[tasklet_id] = (IndexFile*) ptr;
    ptr = ptr + CHUNK_SIZE;
    index_file2[tasklet_id] = (IndexFile*) ptr;
    ptr = ptr + CHUNK_SIZE;
    chunks_offset[tasklet_id] = ptr;
    index_file[tasklet_id] = index_file1[tasklet_id];

#ifdef PINNED_CHUNK_HEADERS
    wr_chunk_headers1[tasklet_id] = /* TODO: update 512 */
                (ChunkHeader*) mem_alloc(512 * sizeof(ChunkHeader));
    wr_chunk_headers2 =
                (ChunkHeader*) mem_alloc(512 * sizeof(ChunkHeader));
    wr_chunk_headers[tasklet_id] = wr_chunk_headers1[tasklet_id];
#endif

    for (uint32_t i = 0; i < num_chunk_hdrs[tasklet_id]; i++) {
        uint32_t cid = alloc_chunk(tasklet_id);
        /* TODO: store ids in WRAM and update all headers first */
        Bucket* chunk = (Bucket*) chunk_ptr(cid, tasklet_id);

        Bucket* wram_bucket = (Bucket*) wr_buffer[tasklet_id];
        for (uint32_t j = 0; j < BUCKETS_PER_CHUNK; j++) {
            wram_bucket->header.bitmap = (uint16_t) 0xC000;
            mram_write(wram_bucket, (__mram_ptr void*) &chunk[j], sizeof(Bucket));
        }

#ifdef PINNED_CHUNK_HEADERS
        wr_chunk_headers1[tasklet_id][i].local_depth = INIT_GLOBAL_DEPTH;
        wr_chunk_headers1[tasklet_id][i].hash_scheme = 1;
        wr_chunk_headers1[tasklet_id][i].cid = cid;
#else
        ChunkHeader* wram_header = (ChunkHeader*) wr_buffer[tasklet_id];
        ChunkHeader* mram_header = &index_file[tasklet_id]->chunk_headers[i];
        wram_header->local_depth = INIT_GLOBAL_DEPTH;
        wram_header->hash_scheme = 1;
        wram_header->cid = cid;
        mram_write(wram_header,
                    (__mram_ptr void*) mram_header, sizeof(ChunkHeader));
#endif
    }
}

PIMValue_t search_bucket(PIMKey_t key, uint8_t fgprint,
                        Bucket* bucket, uint32_t tasklet_id) {
    PIMValue_t ret;
    Bucket* mram_bucket = bucket;
    Bucket* wram_bucket = (Bucket*) wr_buffer[tasklet_id];
    mram_read((__mram_ptr void const*) mram_bucket, wram_bucket, BUCKET_SIZE);

    BucketHeader* bucket_header = &wram_bucket->header;

    uint8_t cmp[ENTRIES_PER_BUCKET_PIM] = {0};
    for (uint32_t i = 0; i < ENTRIES_PER_BUCKET_PIM; i++) {
        cmp[i] = (bucket_header->fingerprints[i] == fgprint) ? 1 : 0;
    }

    for (uint32_t i = 0; i < ENTRIES_PER_BUCKET_PIM; i++) {
        if (cmp[i]) {
            if (wram_bucket->entries[BUCKET_HEADER_SKIP + i].key == key) {
                ret = wram_bucket->entries[BUCKET_HEADER_SKIP + i].val;
                return ret;
            }
        }
    }

    return (PIMValue_t)0;
}

PIMValue_t single_hash_search(PIMKey_t key, HashValue_t hash,
                                uint32_t cid, uint32_t tasklet_id) {
    uint8_t fgprint = fingerprint(hash);
    uint32_t bid = bucket_id(hash);
    Bucket* chunk = (Bucket*) chunk_ptr(cid, tasklet_id);
    Bucket* bucket = &chunk[bid];

    PIMValue_t val = search_bucket(key, fgprint, bucket, tasklet_id);
    return val;
}

PIMValue_t search(PIMKey_t key, uint32_t tasklet_id) {
    PIMValue_t ret;
    HashValue_t hash_val = murmur64(key);
    uint32_t pos = chunk_hdr_pos(hash_val, global_depth[tasklet_id]);

#ifdef PINNED_CHUNK_HEADERS
    uint32_t cid = wr_chunk_headers[tasklet_id][pos].cid;
#else
    ChunkHeader* mram_hdr = &index_file[tasklet_id]->chunk_headers[pos];
    ChunkHeader* hdr = (ChunkHeader*)wr_buffer[tasklet_id];
    mram_read((__mram_ptr const void*) mram_hdr, hdr, sizeof(ChunkHeader));
    uint32_t cid = hdr->cid;
#endif

    ret = single_hash_search(key, hash_val, cid, tasklet_id);
    return ret;
}

int initialization_kernel() {
    uint32_t tasklet_id = me();
    initialize_index(tasklet_id);
    return 0;
}

#ifdef THREE_LEVEL_PARTITIONING
int search_kernel() {

    uint32_t tasklet_id = me();

    wr_buffer2[tasklet_id] = mem_alloc(WRAM_BUFFER_SIZE);
    wr_buffer3[tasklet_id] = mem_alloc(WRAM_BUFFER_SIZE);

    if (tasklet_id == 0) {
        prefix = 0;
        wr_hist = (uint32_t*) mem_alloc(NR_TASKLETS * sizeof(uint32_t));
        wr_part_counter = (uint8_t*) mem_alloc(NR_TASKLETS * sizeof(uint8_t));
        wr_part_buffer = (PIMKey_t*) mem_alloc(NR_TASKLETS * 2 * sizeof(PIMKey_t));
        memset(wr_hist, 0x0, NR_TASKLETS * sizeof(uint32_t));
        memset(wr_part_counter, 0x0, NR_TASKLETS * sizeof(uint8_t));
    }
    barrier_wait(&index_barrier);

    uint32_t part_padding_keys = 0;
    uint32_t histogram_offset;
    uint32_t histogram_size;
    PIMKey_t* wram_keys = (PIMKey_t*) wr_buffer3[tasklet_id];
    PIMKey_t* mram_in_keys = (PIMKey_t*)(mram_heap + MRAM_INPUT_OFFSET);
    PIMKey_t* mram_out_keys = (PIMKey_t*)(mram_heap + MRAM_OUTPUT_OFFSET);

    /* histogram sizes */
    for (uint32_t i = tasklet_id * NR_KEYS_PER_WRAM_BUFFER;
                  i < dpu_args.num_keys;
                  i += NR_TASKLETS * NR_KEYS_PER_WRAM_BUFFER) {
        mram_read((__mram_ptr void const*) &mram_in_keys[i],
                                            wram_keys, WRAM_BUFFER_SIZE);
        uint32_t num_keys =
            ((i + NR_KEYS_PER_WRAM_BUFFER) < dpu_args.num_keys) ?
            (NR_KEYS_PER_WRAM_BUFFER) :
            (dpu_args.num_keys - i);

        for (uint32_t j = 0; j < num_keys; j++) {
            PIMKey_t key = wram_keys[j];
            if (key == 0) {
                part_padding_keys++;
                continue;
            }
            uint32_t tid = sdbm(key) % NR_TASKLETS;
            mutex_pool_lock(&partitioning_mtx, tid);
            wr_hist[tid]++;
            mutex_pool_unlock(&partitioning_mtx, tid);
        }
    }
    ((uint32_t*)wr_buffer2[tasklet_id])[0] = part_padding_keys;
    barrier_wait(&index_barrier);

    /* prefix sums */
    if (tasklet_id == 0) {
        uint32_t val;
        for (uint32_t i = 0; i < NR_TASKLETS; i++) {
            val = wr_hist[i];
            wr_hist[i] = prefix;
            prefix += val;
        }
    }
    barrier_wait(&index_barrier);

    histogram_offset = wr_hist[tasklet_id];
    histogram_size = (tasklet_id == (NR_TASKLETS - 1)) ?
                     (prefix - wr_hist[tasklet_id]) :
                     (wr_hist[tasklet_id + 1] - wr_hist[tasklet_id]);
    barrier_wait(&index_barrier);

    /* tasklet mapping offsets */
    for (uint32_t i = tasklet_id * NR_KEYS_PER_WRAM_BUFFER;
                  i < dpu_args.num_keys;
                  i += NR_TASKLETS * NR_KEYS_PER_WRAM_BUFFER) {
        mram_read((__mram_ptr void const*) &mram_in_keys[i],
                                            wram_keys, WRAM_BUFFER_SIZE);
        uint32_t num_keys =
            ((i + NR_KEYS_PER_WRAM_BUFFER) < dpu_args.num_keys) ?
            (NR_KEYS_PER_WRAM_BUFFER) :
            (dpu_args.num_keys - i);

        for (uint32_t j = 0; j < num_keys; j++) {
            PIMKey_t key = wram_keys[j];
            if (key == 0) {
                continue;
            }
            uint32_t tid = sdbm(key) % NR_TASKLETS;
            mutex_pool_lock(&partitioning_mtx, tid);
            uint32_t mram_offset = 2 * wr_hist[tid];
            wr_hist[tid]++;
            wr_part_buffer[2 * tid] = key;
            wr_part_buffer[2 * tid + 1] = 0;
            mram_write(&wr_part_buffer[2 * tid],
                    (__mram_ptr void*) &mram_out_keys[mram_offset], 2 * KEY_SIZE);
            mutex_pool_unlock(&partitioning_mtx, tid);
        }
    }
    barrier_wait(&index_barrier);

    /* search */
    uint32_t found = 0;
    uint32_t not_found = 0;
    uint32_t padkeys = 0;
    PIMKey_t* mram_keys = &mram_out_keys[2 * histogram_offset];
    PIMKey_t* res = &mram_in_keys[2 * histogram_offset];

    for (uint32_t i = 0; i < (2 * histogram_size); i += NR_KEYS_PER_WRAM_BUFFER) {
        mram_read((__mram_ptr const void*) &mram_keys[i],
                                            wram_keys, WRAM_BUFFER_SIZE);
        uint32_t num_keys = ((i + NR_KEYS_PER_WRAM_BUFFER) < (2 * histogram_size)) ?
                            (NR_KEYS_PER_WRAM_BUFFER) :
                            ((2 * histogram_size) - i);

        for (uint32_t j = 0; j < num_keys; j += 2) {
            if (wram_keys[j] == 0) {
                padkeys++;
                continue;
            }
            PIMValue_t val = search(wram_keys[j], tasklet_id);
            if (val == DEFAULT_VALUE) {
                found++;
            }
            else {
                not_found++;
            }

            wram_keys[j + 1] = val;
        }

        mram_write(wram_keys, (__mram_ptr void*) &res[i], num_keys * KEY_SIZE);
    }

    ((uint32_t*)wr_buffer2[tasklet_id])[1] = 2 * histogram_size;
    barrier_wait(&index_barrier);
    if (tasklet_id == 0) {
        uint32_t num_padkeys = 0;
        uint32_t num_skeys = 0;
        for (uint32_t i = 0; i < NR_TASKLETS; i++) {
            num_padkeys += ((uint32_t*)wr_buffer2[i])[0];
            num_skeys += ((uint32_t*)wr_buffer2[i])[1];
        }

        if (wr_hist[NR_TASKLETS - 1] != (dpu_args.num_keys - num_padkeys)) {
            dpu_args.kret = count_mismatch;
            exit(EXIT_FAILURE);
        }
        // printf("num_keys: %u | num_found %u | num_not_found %u | num_padkeys: %u\n",
        //             dpu_args.num_keys, num_found, num_not_found, num_padkeys);
        dpu_args.num_keys = num_skeys;
        dpu_args.padding = num_padkeys;
    }

    return 0;
}

#else

int search_kernel() {

    uint32_t tasklet_id = me();

    wr_buffer2[tasklet_id] = mem_alloc(WRAM_BUFFER_SIZE);
    wr_buffer3[tasklet_id] = mem_alloc(WRAM_BUFFER_SIZE);

    uint32_t found = 0;
    uint32_t not_found = 0;
    uint32_t keys_per_xfer = NR_KEYS_PER_WRAM_BUFFER / 2;
    PIMKey_t* wram_keys = (PIMKey_t*) wr_buffer3[tasklet_id];
    PIMKey_t* mram_keys = (PIMKey_t*)(mram_heap + MRAM_INPUT_OFFSET);
    PIMKey_t* mram_res = (PIMKey_t*)(mram_heap + MRAM_OUTPUT_OFFSET);

    __dma_aligned uint32_t wram_res[2];
    for (uint32_t i = 0; i < dpu_args.num_keys; i += keys_per_xfer) {
        mram_read((__mram_ptr const void*) &mram_keys[i],
                                            wram_keys, WRAM_BUFFER_SIZE);
        uint32_t num_keys = ((i + keys_per_xfer) < dpu_args.num_keys) ?
                            (keys_per_xfer) :
                            (dpu_args.num_keys - i);

        for (uint32_t j = 0; j < num_keys; j++) {
            PIMKey_t key = wram_keys[j];
            uint32_t tid = key_to_tasklet_hash(key) % NR_TASKLETS;
            if (tid != tasklet_id) {
                continue;
            }
            if (key == 0) {
                wram_res[0] = 0;
                wram_res[1] = 0;
                mram_write(wram_res,
                    (__mram_ptr void*) &mram_res[2 * (i + j)], 2 * KEY_SIZE);
                continue;
            }
            PIMValue_t val = search(key, tasklet_id);
            if (val == DEFAULT_VALUE) {
                found++;
            }
            else {
                not_found++;
            }

            wram_res[0] = key;
            wram_res[1] = val;
            mram_write(wram_res,
                (__mram_ptr void*) &mram_res[2 * (i + j)], 2 * KEY_SIZE);
        }
    }
    // printf("Tasklet: %u | Found: %u\n", tasklet_id, found);

    *((uint32_t*) wr_buffer2[tasklet_id]) = found;
    *((uint32_t*) wr_buffer[tasklet_id]) = not_found;
    barrier_wait(&index_barrier);
    if (tasklet_id == 0) {
        uint32_t num_found = 0;
        uint32_t num_not_found = 0;
        for (uint32_t i = 0; i < NR_TASKLETS; i++) {
            num_found += *((uint32_t*) wr_buffer2[i]);
            num_not_found += *((uint32_t*) wr_buffer[i]);
        }
        // printf("Tasklet: %u | Found: %u | Not found: %u\n",
        //         tasklet_id, num_found, num_not_found);
        if (num_not_found == 0) {
            dpu_args.kret = exec_success;
        }
        else {
            dpu_args.kret = count_mismatch;
        }
    }

    return 0;
}
#endif

int mapping_kernel() {

    uint32_t tasklet_id = me();

    if (tasklet_id == 0) {
        prefix = 0;
        wr_hist = (uint32_t*) mem_alloc(NR_DPUS * sizeof(uint32_t));
        wr_part_buffer = (PIMKey_t*) mem_alloc(NR_DPUS * 2 * sizeof(PIMKey_t));
        wr_part_counter = (uint8_t*) mem_alloc(NR_DPUS * sizeof(uint8_t));
    }
    barrier_wait(&index_barrier);

    for (uint32_t i = tasklet_id; i < NR_DPUS; i += NR_TASKLETS) {
        wr_hist[i] = 0;
    }
    for (uint32_t i = tasklet_id; i < NR_DPUS; i += NR_TASKLETS) {
        wr_part_counter[i] = 0;
    }
    barrier_wait(&index_barrier);

    uint32_t padding_len = 0;
    uint32_t keys_per_buffer = WRAM_BUFFER_SIZE / sizeof(PIMKey_t);
    PIMKey_t* wr_keys = (PIMKey_t*) wr_buffer[tasklet_id];
    PIMKey_t* mr_inkeys = (PIMKey_t*)((char*)mram_heap + MRAM_INPUT_OFFSET);
    PIMKey_t* mr_outkeys =
                (PIMKey_t*)((char*)mram_heap + MRAM_OUTPUT_OFFSET);
    uint32_t* mr_hist = (uint32_t*)((char*)mram_heap + MRAM_INPUT_OFFSET);

    for (uint32_t i = tasklet_id * keys_per_buffer;
                  i < dpu_args.num_keys; i += NR_TASKLETS * keys_per_buffer) {
        mram_read((__mram_ptr void const*) &mr_inkeys[i], wr_keys, WRAM_BUFFER_SIZE);
        uint32_t num_keys = ((i + keys_per_buffer) < dpu_args.num_keys) ?
                            (keys_per_buffer) :
                            (dpu_args.num_keys - i);
        for (uint32_t j = 0; j < num_keys; j++) {
            PIMKey_t key = wr_keys[j];
            uint32_t part = superfast(key) % NR_DPUS;
            mutex_pool_lock(&partitioning_mtx, part);
            wr_hist[part]++;
            mutex_pool_unlock(&partitioning_mtx, part);
        }
    }
    barrier_wait(&index_barrier);

    /* compute prefix sums */
    if (tasklet_id == 0) {
        uint32_t val;
        uint32_t padding;
        for (uint32_t i = 0; i < NR_DPUS; i++) {
            val = wr_hist[i];
            padding = val % 2;
            padding_len += padding;
            val += padding;
            wr_hist[i] = prefix;
            prefix += val;
        }
    }
    barrier_wait(&index_barrier);

    for (uint32_t i = tasklet_id * keys_per_buffer;
                  i < dpu_args.num_keys; i += NR_TASKLETS * keys_per_buffer) {
        mram_read((__mram_ptr void const*) &mr_inkeys[i], wr_keys, WRAM_BUFFER_SIZE);
        uint32_t num_keys = ((i + keys_per_buffer) < dpu_args.num_keys) ?
                            (keys_per_buffer) :
                            (dpu_args.num_keys - i);
        for (uint32_t j = 0; j < num_keys; j++) {
            PIMKey_t key = wr_keys[j];
            uint32_t part = superfast(key) % NR_DPUS;
            uint32_t mr_offs;
            mutex_pool_lock(&partitioning_mtx, part);
            if (wr_part_counter[part] == 0) {
                wr_part_buffer[2 * part] = key;
                wr_part_counter[part] = 1;
                mutex_pool_unlock(&partitioning_mtx, part);
            }
            else {
                wr_part_buffer[2 * part + 1] = key;
                mr_offs = wr_hist[part];
                wr_hist[part] += 2;
                wr_part_counter[part] = 0;
                mutex_pool_unlock(&partitioning_mtx, part);
                mram_write(&wr_part_buffer[2 * part],
                        (__mram_ptr void*) &mr_outkeys[mr_offs], 2 * KEY_SIZE);
            }
        }
    }
    barrier_wait(&index_barrier);

    for (uint32_t part = tasklet_id; part < NR_DPUS; part += NR_TASKLETS) {
        if (wr_part_counter[part] == 1) {
            uint32_t mr_offs = wr_hist[part];
            wr_part_buffer[2 * part + 1] = 0; /* padding key */
            mram_write(&wr_part_buffer[2 * part],
                (__mram_ptr void*) &mr_outkeys[mr_offs], 2 * KEY_SIZE);
            wr_hist[part] += 2;
            wr_part_counter[part] = 0;
        }
    }
    barrier_wait(&index_barrier);

    uint32_t hist_elem_per_buffer = WRAM_BUFFER_SIZE / sizeof(uint32_t);
    for (uint32_t i = tasklet_id * hist_elem_per_buffer;
                  i < NR_DPUS; i += NR_TASKLETS * hist_elem_per_buffer) {
        mram_write(&wr_hist[i], (__mram_ptr void*) &mr_hist[i], WRAM_BUFFER_SIZE);
    }
    barrier_wait(&index_barrier);

    if (tasklet_id == 0) {
        if ((wr_hist[NR_DPUS - 1] - padding_len) == dpu_args.num_keys) {
            dpu_args.kret = exec_success;
        }
        else {
            dpu_args.kret = count_mismatch;
        }
    }

    return 0;
}

int kvmapping_kernel() {

    uint32_t tasklet_id = me();

    if (tasklet_id == 0) {
        prefix = 0;
        wr_hist = (uint32_t*) mem_alloc(NR_DPUS * sizeof(uint32_t));
    }
    barrier_wait(&index_barrier);

    for (uint32_t i = tasklet_id; i < NR_DPUS; i += NR_TASKLETS) {
        wr_hist[i] = 0;
    }
    barrier_wait(&index_barrier);

    uint32_t keys_per_buffer = WRAM_BUFFER_SIZE / sizeof(PIMKey_t);
    PIMKey_t* wr_keys = (PIMKey_t*) wr_buffer[tasklet_id];
    PIMKey_t* mr_inkeys = (PIMKey_t*)((char*)mram_heap + MRAM_INPUT_OFFSET);
    PIMKey_t* mr_outkeys = (PIMKey_t*)((char*)mram_heap + MRAM_OUTPUT_OFFSET);
    uint32_t* mr_hist = (uint32_t*)((char*)mram_heap + MRAM_INPUT_OFFSET);

    for (uint32_t i = tasklet_id * keys_per_buffer;
                  i < (2 * dpu_args.num_keys); i += NR_TASKLETS * keys_per_buffer) {
        mram_read((__mram_ptr void const*) &mr_inkeys[i], wr_keys, WRAM_BUFFER_SIZE);
        uint32_t num_keys = ((i + keys_per_buffer) < (2 * dpu_args.num_keys)) ?
                            (keys_per_buffer) :
                            ((2 * dpu_args.num_keys) - i);
        for (uint32_t j = 0; j < num_keys; j += 2) {
            PIMKey_t key = wr_keys[j];
            uint32_t part = superfast(key) % NR_DPUS;
            mutex_pool_lock(&partitioning_mtx, part);
            wr_hist[part]++;
            mutex_pool_unlock(&partitioning_mtx, part);
        }
    }
    barrier_wait(&index_barrier);

    /* compute prefix sums */
    if (tasklet_id == 0) {
        uint32_t val;
        for (uint32_t i = 0; i < NR_DPUS; i++) {
            val = wr_hist[i];
            wr_hist[i] = prefix;
            prefix += val;
        }
    }
    barrier_wait(&index_barrier);

    for (uint32_t i = tasklet_id * keys_per_buffer;
                  i < (2 * dpu_args.num_keys); i += NR_TASKLETS * keys_per_buffer) {
        mram_read((__mram_ptr void const*) &mr_inkeys[i], wr_keys, WRAM_BUFFER_SIZE);
        uint32_t num_keys = ((i + keys_per_buffer) < (2 * dpu_args.num_keys)) ?
                            (keys_per_buffer) :
                            ((2* dpu_args.num_keys) - i);
        for (uint32_t j = 0; j < num_keys; j += 2) {
            PIMKey_t key = wr_keys[j];
            uint32_t part = superfast(key) % NR_DPUS;
            mutex_pool_lock(&partitioning_mtx, part);
            uint32_t mr_offs = wr_hist[part];
            wr_hist[part]++;
            mram_write(&wr_keys[j],
                    (__mram_ptr void*) &mr_outkeys[2 * mr_offs], 2 * KEY_SIZE);
            mutex_pool_unlock(&partitioning_mtx, part);
        }
    }
    barrier_wait(&index_barrier);

    uint32_t hist_elem_per_buffer = WRAM_BUFFER_SIZE / sizeof(uint32_t);
    for (uint32_t i = tasklet_id * hist_elem_per_buffer;
                  i < NR_DPUS; i += NR_TASKLETS * hist_elem_per_buffer) {
        mram_write(&wr_hist[i], (__mram_ptr void*) &mr_hist[i], WRAM_BUFFER_SIZE);
    }
    barrier_wait(&index_barrier);

    if (tasklet_id == 0) {
        if (wr_hist[NR_DPUS - 1] == dpu_args.num_keys) {
            dpu_args.kret = exec_success;
        }
        else {
            dpu_args.kret = count_mismatch;
        }
    }

    return 0;
}

int mem_utilization_kernel() {

    uint32_t tasklet_id = me();

    uint32_t alloced_chunks = 0;
    uint32_t unalloced_chunks = 0;
    uint32_t chunks_cnt = 0;
    uint32_t buckets_cnt = 0;
    uint32_t keys_cnt = 0;
    uint32_t prev_cid = (uint32_t)(-1);

    for (uint32_t i = 0; i < BITMAP_LEN_PER_TASKLET; i++) {
        uint32_t byte = wr_chunk_bitmap[tasklet_id][i];
        if (byte == (uint32_t)0xffffffff) {
            unalloced_chunks += 32;
        }
        else {
            for (uint32_t j = 0; j < 32; j++) {
                uint32_t mask = byte & (1u << (31 - j));
                if (mask) {
                    unalloced_chunks++;
                }
                else {
                    alloced_chunks++;
                }
            }
        }
    }
    /*printf("Allocated chunks: %u\n", alloced_chunks);
    printf("Unallocated chunks: %u\n", unalloced_chunks);
    printf("Chunk headers: %u\n", num_chunk_hdrs[tasklet_id]);
    printf("Next free chunk id: %lu\n", next_free_chunk_id[tasklet_id]);*/

#ifdef PINNED_CHUNK_HEADERS
        uint32_t* wr_state = (uint32_t*)wr_buffer[tasklet_id];
        /* logging space for chunk headers */
        ChunkHeader* logged_chunk_headers =
            (ChunkHeader*)((char*)mram_heap +
                MRAM_LOG_OFFSET + tasklet_id * 8 * KiB + 128);
#endif

    ChunkHeader hdr;
    ChunkHeader* wr_chunk_hdrs = &hdr;
    /* ChunkHeader* wr_chunk_hdrs = (ChunkHeader*) wr_buffer2[tasklet_id]; */
    Bucket* wr_buckets = (Bucket*) wr_buffer[tasklet_id];
    /* uint32_t chnk_hdrs_per_xfer = WRAM_BUFFER_SIZE / sizeof(ChunkHeader); */
    uint32_t chnk_hdrs_per_xfer = 1;
    uint32_t buckets_per_xfer = WRAM_BUFFER_SIZE / BUCKET_SIZE;

    for (uint32_t i = 0; i < num_chunk_hdrs[tasklet_id]; i += chnk_hdrs_per_xfer) {
        uint32_t headers = ((i + chnk_hdrs_per_xfer) < num_chunk_hdrs[tasklet_id]) ?
                           (chnk_hdrs_per_xfer) :
                           (num_chunk_hdrs[tasklet_id] - i);

#ifdef PINNED_CHUNK_HEADERS
        ChunkHeader* mr_chunk_hdrs = &logged_chunk_headers[i];
#else
        ChunkHeader* mr_chunk_hdrs = &index_file[tasklet_id]->chunk_headers[i];
#endif
        /* mram_read((__mram_ptr const void*) mr_chunk_hdrs, wr_chunk_hdrs, WRAM_BUFFER_SIZE); */
        mram_read((__mram_ptr const void*) mr_chunk_hdrs,
                                            wr_chunk_hdrs, sizeof(ChunkHeader));

        for (uint32_t j = 0; j < headers; j++) {
            uint32_t cid = wr_chunk_hdrs[j].cid;
            if (cid == prev_cid) {
                continue;
            }

            Bucket* chunk = (Bucket*) chunk_ptr(cid, tasklet_id);
            for (uint32_t k = 0; k < BUCKETS_PER_CHUNK; k += buckets_per_xfer) {
                mram_read((__mram_ptr const void*) &chunk[k],
                                                    wr_buckets, WRAM_BUFFER_SIZE);
                uint32_t buckets = ((k + buckets_per_xfer) < BUCKETS_PER_CHUNK) ?
                                   (buckets_per_xfer) :
                                   (BUCKETS_PER_CHUNK - k);

                for (uint32_t l = 0; l < buckets; l++) {
                    Bucket* bucket = &wr_buckets[l];
                    uint16_t bmap = bucket->header.bitmap;
                    if (bmap != (uint16_t)0xc000) {
                        uint32_t slots;
                        uint32_t bmap32 = (uint32_t)bmap << 16;
                        __builtin_cao_rr(slots, bmap32);
                        keys_cnt += (slots - 2);
                        buckets_cnt++;
                    }
                }
            }
            prev_cid = cid;
            chunks_cnt++;
        }
    }
    /*printf("Chunks: %u\n", chunks_cnt);
    printf("Buckets: %u\n", buckets_cnt);
    printf("Keys: %u\n", keys_cnt);*/
    uint32_t* counts = (uint32_t*) wr_buffer[tasklet_id];
    counts[0] = chunks_cnt;
    counts[1] = buckets_cnt;
    counts[2] = keys_cnt;
    barrier_wait(&index_barrier);
    if (tasklet_id == 0) {
        uint32_t total_chunks_cnt = 0;
        uint32_t total_buckets_cnt = 0;
        uint32_t total_keys_cnt = 0;
        for (uint32_t i = 0; i < NR_TASKLETS; i++) {
            uint32_t* cnts = (uint32_t*) wr_buffer[i];
            total_chunks_cnt += cnts[0];
            total_buckets_cnt += cnts[1];
            total_keys_cnt += cnts[2];
        }
        dpu_args.num_chunks = total_chunks_cnt;
        dpu_args.num_buckets = total_buckets_cnt;
        dpu_args.num_keys = total_keys_cnt;
    }

    return 0;
}

int logging_kernel() {

    uint32_t tasklet_id = me();

    uint32_t* wr_state = (uint32_t*)wr_buffer[tasklet_id];
    uint32_t* mr_state = (uint32_t*)((char*)mram_heap +
                                MRAM_LOG_OFFSET + tasklet_id * 8 * KiB);
    uint32_t i = 0;
    wr_state[i++] = (uint32_t)(index_file[tasklet_id]);
    wr_state[i++] = (uint32_t)(index_file1[tasklet_id]);
    wr_state[i++] = (uint32_t)(index_file2[tasklet_id]);
    wr_state[i++] = (uint32_t)(chunks_offset[tasklet_id]);
    wr_state[i++] = (uint32_t)(global_depth[tasklet_id]);
    wr_state[i++] = num_chunk_hdrs[tasklet_id];
    wr_state[i++] = next_free_chunk_id[tasklet_id];
    wr_state[i++] = index_file_lock[tasklet_id];
    for (uint32_t j = 0; j < BITMAP_LEN_PER_TASKLET; j++) {
        wr_state[i++] = wr_chunk_bitmap[tasklet_id][j];
    }
    /* actual size: 80 bytes */
    mram_write(wr_state, (__mram_ptr void*) mr_state, 128);

#ifdef PINNED_CHUNK_HEADERS
    if (dpu_args.log_level == 2) {
        mr_state = (uint32_t*)((char*)mr_state + 128);
        ChunkHeader* mr_chunk_headers = (ChunkHeader*) mr_state;
        uint32_t chunk_hdrs_per_buffer = WRAM_BUFFER_SIZE / sizeof(ChunkHeader);
        for (uint32_t i = 0; i < 512; i += chunk_hdrs_per_buffer) {
            /* actual size: 4 * KiB */
            mram_write(&wr_chunk_headers[i],
                (__mram_ptr void*) &mr_chunk_headers[i], WRAM_BUFFER_SIZE);
        }
    }
#endif

    return 0;
}

int recovery_kernel() {

    uint32_t tasklet_id = me();

    if (tasklet_id == 0) {
        mem_reset();

        wr_buffer = (void**) mem_alloc(NR_TASKLETS * sizeof(void*));
        wr_buffer2 = (void**) mem_alloc(NR_TASKLETS * sizeof(void*));
        wr_buffer3 = (void**) mem_alloc(NR_TASKLETS * sizeof(void*));
        wr_chunk_bitmap = (uint32_t**) mem_alloc(NR_TASKLETS * sizeof(uint32_t*));
        index_file = (IndexFile**) mem_alloc(NR_TASKLETS * sizeof(IndexFile*));
        index_file1 = (IndexFile**) mem_alloc(NR_TASKLETS * sizeof(IndexFile*));
        index_file2 = (IndexFile**) mem_alloc(NR_TASKLETS * sizeof(IndexFile*));
        chunks_offset = (char**) mem_alloc(NR_TASKLETS * sizeof(char*));

        global_depth = (uint8_t*) mem_alloc(NR_TASKLETS * sizeof(uint8_t));
        num_chunk_hdrs = (uint32_t*) mem_alloc(NR_TASKLETS * sizeof(uint32_t));
        next_free_chunk_id = (uint32_t*) mem_alloc(NR_TASKLETS * sizeof(uint32_t));
        index_file_lock = (uint32_t*) mem_alloc(NR_TASKLETS * sizeof(uint32_t));
#ifdef PINNED_CHUNK_HEADERS
        if (dpu_args.log_level == 2) {
            wr_chunk_headers =
                    (ChunkHeader**) mem_alloc(NR_TASKLETS * sizeof(ChunkHeader*));
            wr_chunk_headers1 =
                    (ChunkHeader**) mem_alloc(NR_TASKLETS * sizeof(ChunkHeader*));
            wr_chunk_headers2 =
                    (ChunkHeader**) mem_alloc(NR_TASKLETS * sizeof(ChunkHeader*));
        }
#endif
    }
    barrier_wait(&index_barrier);

    wr_buffer[tasklet_id] = mem_alloc(WRAM_BUFFER_SIZE);
    wr_chunk_bitmap[tasklet_id] =
            (uint32_t*) mem_alloc(BITMAP_LEN_PER_TASKLET * sizeof(uint32_t));

    uint32_t* wr_state = (uint32_t*)wr_buffer[tasklet_id];
    uint32_t* mr_state = (uint32_t*)((char*)mram_heap +
                                MRAM_LOG_OFFSET + tasklet_id * 8 * KiB);

    mram_read((__mram_ptr const void*) mr_state, wr_state, 128);

    uint32_t i = 0;
    index_file[tasklet_id] = (IndexFile*)(wr_state[i++]);
    index_file1[tasklet_id] = (IndexFile*)(wr_state[i++]);
    index_file2[tasklet_id] = (IndexFile*)(wr_state[i++]);
    chunks_offset[tasklet_id] = (char*)(wr_state[i++]);
    global_depth[tasklet_id] = (uint8_t)(wr_state[i++]);
    num_chunk_hdrs[tasklet_id] = wr_state[i++];
    next_free_chunk_id[tasklet_id] = wr_state[i++];
    index_file_lock[tasklet_id] = wr_state[i++];
    for (uint32_t j = 0; j < BITMAP_LEN_PER_TASKLET; j++) {
        wr_chunk_bitmap[tasklet_id][j] = wr_state[i++];
    }

#ifdef PINNED_CHUNK_HEADERS
    if (dpu_args.log_level == 2) {
        wr_chunk_headers1[tasklet_id] = /* TODO: update 512 */
                    (ChunkHeader*) mem_alloc(512 * sizeof(ChunkHeader));
        wr_chunk_headers2 =
                    (ChunkHeader*) mem_alloc(512 * sizeof(ChunkHeader));
        wr_chunk_headers[tasklet_id] =
                    (index_file_lock[tasklet_id] & 0x80000000u) ?
                    (wr_chunk_headers2[tasklet_id]) :
                    (wr_chunk_headers1[tasklet_id]);

        mr_state = (uint32_t*)((char*)mr_state + 128);
        ChunkHeader* mr_chunk_headers = (ChunkHeader*) mr_state;
        uint32_t chunk_hdrs_per_buffer = WRAM_BUFFER_SIZE / sizeof(ChunkHeader);
        for (uint32_t i = 0; i < 512; i += chunk_hdrs_per_buffer) {
            mram_read((__mram_ptr const void*) &mr_chunk_headers[i],
                                &wr_chunk_headers[i], WRAM_BUFFER_SIZE);
        }
    }
#endif

    return 0;
}

int (*kernels[NR_KERNELS])() =
    {initialization_kernel, kvmapping_kernel, mapping_kernel,
        search_kernel, mem_utilization_kernel, logging_kernel, recovery_kernel};

int main() {
    return kernels[dpu_args.kernel]();
}
