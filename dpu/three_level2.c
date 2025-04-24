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
uint32_t* index_file_lock; /* [31:] ptr, [30:] lock bit, [29:0] version */

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

static inline uint32_t chunk_hdr_ext_pos(HashValue_t hash, uint8_t new_depth) {
    return hash << (16 + new_depth) >> 30;
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

static inline uint32_t chunk_hdr_ext_pos(HashValue_t hash, uint8_t new_depth) {
    return hash << (30 + new_depth) >> 62;
}

static inline uint8_t fingerprint(HashValue_t hash) {
    return hash >> 56;
}

static inline uint32_t bucket_id(HashValue_t hash) {
    return (hash << 8) >> 54;
}
#endif

static inline void* chunk_ptr(uint32_t cid, uint32_t tasklet_id) {
    return chunks_offset[tasklet_id] + cid * CHUNK_SIZE;
}

static inline uint16_t is_bit_set(uint16_t bitmap, uint32_t slot) {
    return bitmap & (1u << ((ENTRIES_PER_BUCKET_PIM - 1) - slot));
}

static inline uint32_t next_free_slot(uint16_t bitmap) {
    uint32_t mask = ~((uint32_t)bitmap << 16);
    uint32_t slot;
    __builtin_clz_rr(slot, mask);
    return slot;
}

static inline uint16_t set_bitmap(uint16_t bitmap, uint32_t slot) {
    return bitmap | (1u << (15 - slot));
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

void alloc_chunks(uint32_t* cids, uint32_t tasklet_id) {
    uint32_t cnt = 0;
    if (next_free_chunk_id[tasklet_id] > (MAX_NUM_CHUNKS_PER_TASKLET - 1)) {
        printf("No more chunks...\n");
        exit(EXIT_FAILURE);
    }

    do {
        uint32_t old_free_cid = next_free_chunk_id[tasklet_id];
        uint32_t byte_pos = old_free_cid / 32u;
        uint32_t bit_pos, old_val, new_val;
        byte_pos--;

        do {
            byte_pos++;
            old_val = wr_chunk_bitmap[tasklet_id][byte_pos];
        }
        while (old_val == 0);

        uint32_t bit_cnt;
        __builtin_cao_rr(bit_cnt, old_val);
        uint32_t bits = (bit_cnt < (4 - cnt)) ? bit_cnt : (4 - cnt);
        new_val = old_val;
        for (uint32_t i = 0; i < bits; i++) {
            __builtin_clz_rr(bit_pos, new_val);
            new_val = new_val & ~(1u << (31 - bit_pos));
            cids[cnt + i] = bit_pos;
        }

        wr_chunk_bitmap[tasklet_id][byte_pos] = new_val;
        for (uint32_t i = 0; i < bits; i++) {
            cids[cnt + i] += (byte_pos * 32);
        }
        cnt += bits;

        uint32_t last_free_slot = cids[cnt - 1];
        next_free_chunk_id[tasklet_id] = last_free_slot + 1;

        if ((next_free_chunk_id[tasklet_id] > (MAX_NUM_CHUNKS - 1)) && (cnt < 4)) {
            printf("No more chunks\n");
            exit(EXIT_FAILURE);
        }
    }
    while (cnt != 4);
}

void free_chunk(uint32_t cid, uint32_t tasklet_id) {
    uint32_t byte_pos = cid / 32;
    uint32_t bit_pos = cid % 32;

    uint32_t old_val, new_val;
    old_val = wr_chunk_bitmap[tasklet_id][byte_pos];
    new_val = old_val | (1u << (31 - bit_pos));
    wr_chunk_bitmap[tasklet_id][byte_pos] = new_val;

    uint32_t old_next_free_cid = next_free_chunk_id[tasklet_id];
    if (old_next_free_cid < cid) {
        return;
    }
    next_free_chunk_id[tasklet_id] = cid;
}

#ifdef PINNED_CHUNK_HEADERS
void expand_index_file(uint32_t hdr_pos, ChunkHeader* new_chunk_headers,
                                                    uint32_t tasklet_id) {
    ChunkHeader* new_wram_chunk_headers;
    if (index_file_lock[tasklet_id] & 0x80000000u) {
        new_wram_chunk_headers = wr_chunk_headers1[tasklet_id];
    }
    else {
        new_wram_chunk_headers = wr_chunk_headers2[tasklet_id];
    }

    for (uint32_t i = 0; i < num_chunk_hdrs[tasklet_id]; i++) {
        uint32_t j = (i << 2);
        ChunkHeader* wram_new_hdrs = &new_wram_chunk_headers[j];
        for (uint32_t k = 0; k < 4; k++) {
            wram_new_hdrs[k].local_depth =
                        wr_chunk_headers[tasklet_id][i].local_depth;
            wram_new_hdrs[k].hash_scheme =
                        wr_chunk_headers[tasklet_id][i].hash_scheme;
            wram_new_hdrs[k].cid =
                        wr_chunk_headers[tasklet_id][i].cid;
        }
    }

    ChunkHeader* headers = (ChunkHeader*) wr_buffer[tasklet_id];
    mram_read((__mram_ptr const void*) new_chunk_headers,
                                        headers, sizeof(ChunkHeader) * 4);

    uint32_t jj = (hdr_pos << 2);
    ChunkHeader* wram_new_hdrs = &new_wram_chunk_headers[jj];
    for (uint32_t k = 0; k < 4; k++) {
        wram_new_hdrs[k].local_depth = headers[k].local_depth;
        wram_new_hdrs[k].hash_scheme = headers[k].hash_scheme;
        wram_new_hdrs[k].cid = headers[k].cid;
    }

    global_depth[tasklet_id] += 2;
    num_chunk_hdrs[tasklet_id] = (1u << global_depth[tasklet_id]);
    wr_chunk_headers[tasklet_id] = new_wram_chunk_headers;
}

void update_index_file(uint32_t hdr_pos, ChunkHeader* new_chunk_headers,
                                uint8_t old_depth, uint32_t tasklet_id) {
    uint8_t depth_diff = global_depth - old_depth;
    uint32_t start_pos = hdr_pos >> depth_diff << depth_diff;
    uint32_t old_num_chunk_diff = 1u << depth_diff;
    uint32_t new_num_chunk_diff = old_num_chunk_diff >> 2;

    ChunkHeader* headers = (ChunkHeader*) wr_buffer[tasklet_id];
    mram_read((__mram_ptr const void*) new_chunk_headers,
                                        headers, 4 * sizeof(ChunkHeader));

    for (uint32_t i = 0; i < 4; i++) {
        uint32_t j = start_pos + new_num_chunk_diff * i;
        for (uint32_t pos = j; pos < (j + new_num_chunk_diff); pos++) {
            wr_chunk_headers[tasklet_id][pos].cid = headers[i].cid;
            wr_chunk_headers[tasklet_id][pos].hash_scheme = headers[i].hash_scheme;
            wr_chunk_headers[tasklet_id][pos].local_depth = headers[i].local_depth;
        }
    }
}
#else
void expand_index_file(uint32_t hdr_pos, ChunkHeader* new_chunk_headers,
                                                        uint32_t tasklet_id) {
    IndexFile* new_mram_index_file;
    IndexFile* old_mram_index_file = index_file[tasklet_id];
    if (index_file_lock[tasklet_id] & 0x80000000u) {
        new_mram_index_file = index_file1[tasklet_id];
    }
    else {
        new_mram_index_file = index_file2[tasklet_id];
    }

    for (uint32_t i = 0; i < num_chunk_hdrs[tasklet_id]; i++) {
        ChunkHeader* mram_old_hdrs = &old_mram_index_file->chunk_headers[i];
        ChunkHeader* wram_old_hdrs = (ChunkHeader*) wr_buffer2[tasklet_id];
        mram_read((__mram_ptr const void*) mram_old_hdrs,
                                        wram_old_hdrs, sizeof(ChunkHeader));

        ChunkHeader* wram_new_hdrs = (ChunkHeader*) wr_buffer[tasklet_id];
        for (uint32_t k = 0; k < 4; k++) {
            wram_new_hdrs[k].local_depth = wram_old_hdrs->local_depth;
            wram_new_hdrs[k].hash_scheme = wram_old_hdrs->hash_scheme;
            wram_new_hdrs[k].cid = wram_old_hdrs->cid;
        }

        uint32_t j = (i << 2);
        ChunkHeader* mram_new_hdrs = &new_mram_index_file->chunk_headers[j];
        mram_write(wram_new_hdrs,
                    (__mram_ptr void*) mram_new_hdrs, sizeof(ChunkHeader) * 4);
    }

    ChunkHeader* wram_new_hdrs = (ChunkHeader*) wr_buffer[tasklet_id];
    mram_read((__mram_ptr const void*) new_chunk_headers,
                                    wram_new_hdrs, sizeof(ChunkHeader) * 4);

    uint32_t jj = (hdr_pos << 2);
    ChunkHeader* mram_new_hdrs = &new_mram_index_file->chunk_headers[jj];
    mram_write(wram_new_hdrs,
                    (__mram_ptr void*) mram_new_hdrs, sizeof(ChunkHeader) * 4);

    global_depth[tasklet_id] += 2;
    num_chunk_hdrs[tasklet_id] = (1u << global_depth[tasklet_id]);
    index_file[tasklet_id] = new_mram_index_file;
}

void update_index_file(uint32_t hdr_pos, ChunkHeader* new_chunk_headers,
                                    uint8_t old_depth, uint32_t tasklet_id) {
    uint8_t depth_diff = global_depth[tasklet_id] - old_depth;
    uint32_t start_pos = hdr_pos >> depth_diff << depth_diff;
    uint32_t old_num_chunk_diff = 1u << depth_diff;
    uint32_t new_num_chunk_diff = old_num_chunk_diff >> 2;

    ChunkHeader* wram_old_chunk_hdrs = (ChunkHeader*) wr_buffer2[tasklet_id];
    ChunkHeader* wram_new_chunk_hdrs = (ChunkHeader*) wr_buffer[tasklet_id];
    mram_read((__mram_ptr const void*) new_chunk_headers,
                                wram_new_chunk_hdrs, 4 * sizeof(ChunkHeader));

    uint32_t set_chunks = WRAM_BUFFER_SIZE / sizeof(ChunkHeader);
    for (uint32_t i = 0; i < 4; i++) {
        uint32_t pos = start_pos + new_num_chunk_diff * i;
        for (uint32_t j = pos; j < (pos + new_num_chunk_diff); j += set_chunks) {
            uint32_t chunks = ((j + set_chunks) < (pos + new_num_chunk_diff)) ?
                              (set_chunks) :
                              ((pos + new_num_chunk_diff) - j);

            for (uint32_t k = 0; k < chunks; k++) {
                wram_old_chunk_hdrs[k].cid =
                                        wram_new_chunk_hdrs[i].cid;
                wram_old_chunk_hdrs[k].hash_scheme =
                                        wram_new_chunk_hdrs[i].hash_scheme;
                wram_old_chunk_hdrs[k].local_depth =
                                        wram_new_chunk_hdrs[i].local_depth;
            }

            ChunkHeader* mram_old_chunk_hdrs =
                                        &index_file[tasklet_id]->chunk_headers[j];
            mram_write(wram_old_chunk_hdrs, (__mram_ptr void*) mram_old_chunk_hdrs,
                                        chunks * sizeof(ChunkHeader));
        }
    }
}
#endif

static inline bool single_hash_reassign_bucket_entries(Bucket* old_bucket,
                uint8_t new_depth, uint32_t* new_cids, uint32_t tasklet_id) {
    Bucket* wram_old_bucket = (Bucket*) wr_buffer2[tasklet_id]; /* pin in WRAM */
    mram_read((__mram_ptr const void*) old_bucket, wram_old_bucket, BUCKET_SIZE);

    uint32_t bid;
    HashValue_t hash_val;
    uint32_t new_cid;
    BucketEntry* old_entry;
    for (uint32_t i = 0; i < ENTRIES_PER_BUCKET_PIM; i++) {
        if (is_bit_set(wram_old_bucket->header.bitmap, i)) {
            old_entry = &wram_old_bucket->entries[BUCKET_HEADER_SKIP + i];
            hash_val = murmur64(old_entry->key);
            bid = bucket_id(hash_val);
            new_cid = new_cids[chunk_hdr_ext_pos(hash_val, new_depth)];

            Bucket* new_chunk = (Bucket*) chunk_ptr(new_cid, tasklet_id);
            Bucket* wram_new_bucket = (Bucket*) wr_buffer[tasklet_id];
            mram_read((__mram_ptr const void*) &new_chunk[bid],
                                        wram_new_bucket, sizeof(Bucket));
            uint32_t slot = next_free_slot(wram_new_bucket->header.bitmap);

            if (slot == (BUCKET_HEADER_SKIP + ENTRIES_PER_BUCKET_PIM)) {
                return false;
            }
            else {
                wram_new_bucket->entries[slot] = *old_entry;
                uint8_t fgprint = wram_old_bucket->header.fingerprints[i];
                wram_new_bucket->header.fingerprints[slot -
                                                     BUCKET_HEADER_SKIP] = fgprint;
                wram_new_bucket->header.bitmap =
                                set_bitmap(wram_new_bucket->header.bitmap, slot);
                mram_write(wram_new_bucket,
                                (__mram_ptr void*) &new_chunk[bid], sizeof(Bucket));
            }

        }
    }
    return true;
}

ChunkHeader* split_chunk(uint32_t old_cid, uint8_t old_local_depth,
                                                    uint32_t tasklet_id) {
    uint32_t new_cids[4];
    alloc_chunks(new_cids, tasklet_id);
    ChunkHeader* new_hdrs = (ChunkHeader*) wr_buffer[tasklet_id];
    /*uint32_t* new_cids = (uint32_t*) cache;
    alloc_chunks(new_cids);
    ChunkHeader* new_hdrs = (ChunkHeader*) &new_cids[4];*/
    uint8_t new_depth = old_local_depth + 2;

    for (uint32_t i = 0; i < 4; i++) {
        new_hdrs[i].local_depth = new_depth;
        new_hdrs[i].hash_scheme = 1;
        new_hdrs[i].cid = new_cids[i];
    }

#ifdef PINNED_CHUNK_HEADERS
    /* TODO: modify */
    ChunkHeader* mram_new_hdrs = &index_file1[tasklet_id]->chunk_headers[512 + 16];
#else
    /* TODO: modify */
    ChunkHeader* mram_new_hdrs = &index_file[tasklet_id]->chunk_headers[512 + 16];
#endif
    mram_write(new_hdrs, (__mram_ptr void*) mram_new_hdrs, sizeof(ChunkHeader) * 4);

    for (uint32_t i = 0; i < 4; i++) {
        uint32_t new_cid = new_cids[i];
        Bucket* new_chunk = (Bucket*) chunk_ptr(new_cid, tasklet_id);
        Bucket* new_bucket = (Bucket*) wr_buffer[tasklet_id];

        for (uint32_t j = 0; j < BUCKETS_PER_CHUNK; j++) {
            new_bucket->header.bitmap = (uint16_t) 0xc000;
            mram_write(new_bucket, (__mram_ptr void*) &new_chunk[j], sizeof(Bucket));
        }
    }

    Bucket* old_chunk = (Bucket*) chunk_ptr(old_cid, tasklet_id);
    for (uint32_t i = 0; i < BUCKETS_PER_CHUNK; i++) {
        Bucket* old_bucket = &old_chunk[i];
        if (!single_hash_reassign_bucket_entries(old_bucket, new_depth,
                                                 new_cids, tasklet_id)) {
            ; /* TODO */
        }
    }

    return mram_new_hdrs;
}

static inline bool key_exists(Bucket* bucket, PIMKey_t key, uint8_t fgprint) {
    BucketHeader* header = &bucket->header;

    uint8_t cmp[ENTRIES_PER_BUCKET_PIM] = {0};
    for (uint32_t i = 0; i < ENTRIES_PER_BUCKET_PIM; i++) {
        cmp[i] = ((header->bitmap | (1u << (13 - i))) &&
                    (header->fingerprints[i] == fgprint)) ? 1 : 0;
    }

    for (uint32_t i = 0; i < ENTRIES_PER_BUCKET_PIM; i++) {
        if (cmp[i]) {
            if (bucket->entries[BUCKET_HEADER_SKIP + i].key == key) {
                return true;
            }
        }
    }

    return false;
}

static inline void insert_bucket_entry(Bucket* bucket, PIMKey_t key,
                                        PIMValue_t val, uint8_t fgprint) {
    BucketHeader* buc_header = &bucket->header;
    uint32_t slot = next_free_slot(buc_header->bitmap);
    bucket->entries[slot].key = key;
    bucket->entries[slot].val = val;

    buc_header->fingerprints[slot - BUCKET_HEADER_SKIP] = fgprint;
    buc_header->bitmap = set_bitmap(buc_header->bitmap, slot);
}

OpRet single_hash_insert(PIMKey_t key, PIMValue_t val,
            HashValue_t hash, uint32_t cid, uint32_t tasklet_id) {

    uint32_t bid = bucket_id(hash);
    Bucket* chunk = (Bucket*) chunk_ptr(cid, tasklet_id);
    Bucket* mram_bucket = &chunk[bid];
    Bucket* wram_bucket = (Bucket*) wr_buffer[tasklet_id];
    mram_read((__mram_ptr void const*) mram_bucket, wram_bucket, BUCKET_SIZE);

    uint8_t fgprint = fingerprint(hash);
    if (key_exists(wram_bucket, key, fgprint)) {
        return duplicate_key;
    }

    uint32_t bit_cnt;
    uint32_t buc_bitmap = (uint32_t) wram_bucket->header.bitmap;
    __builtin_cao_rr(bit_cnt, buc_bitmap);

    if (bit_cnt == (BUCKET_HEADER_SKIP + ENTRIES_PER_BUCKET_PIM)) {
        return bucket_full;
    }
    else {
        insert_bucket_entry(wram_bucket, key, val, fgprint);
        mram_write(wram_bucket, (__mram_ptr void*) mram_bucket, BUCKET_SIZE);
        return entry_inserted;
    }
}

OpRet insert(PIMKey_t key, PIMValue_t val, uint32_t tasklet_id) {
    OpRet ret;
    HashValue_t hash_val = murmur64(key);
REDO:
    ;
    uint32_t pos = chunk_hdr_pos(hash_val, global_depth[tasklet_id]);
#ifdef PINNED_CHUNK_HEADERS
    uint32_t cid = wr_chunk_headers[pos].cid;
    uint8_t depth = wr_chunk_headers[pos].local_depth;
#else
    ChunkHeader* mram_header = &index_file[tasklet_id]->chunk_headers[pos];
    ChunkHeader* header = (ChunkHeader*)wr_buffer[tasklet_id];
    mram_read((__mram_ptr const void*) mram_header, header, sizeof(ChunkHeader));
    uint32_t cid = header->cid;
    uint8_t depth = header->local_depth;
#endif

    ret = single_hash_insert(key, val, hash_val, cid, tasklet_id);

    if (ret == bucket_full) {
        /* TODO */;
        ChunkHeader* new_chunk_headers = split_chunk(cid, depth, tasklet_id);
        if (depth < global_depth[tasklet_id]) {
            update_index_file(pos, new_chunk_headers, depth, tasklet_id);
        }
        else {
            expand_index_file(pos, new_chunk_headers, tasklet_id);
            if (index_file_lock[tasklet_id] & 0x80000000) {
                index_file_lock[tasklet_id] &= 0x7FFFFFFF;
            }
            else {
                index_file_lock[tasklet_id] |= 0x80000000;
            }
        }
        free_chunk(cid, tasklet_id);
        goto REDO;
    }
    else {
        return ret;
    }
}

#ifdef THREE_LEVEL_PARTITIONING
int insert_kernel() {

    uint32_t tasklet_id = me();

    wr_buffer2[tasklet_id] = mem_alloc(WRAM_BUFFER_SIZE);
    wr_buffer3[tasklet_id] = mem_alloc(WRAM_BUFFER_SIZE);

    if (tasklet_id == 0) {
        prefix = 0;
        wr_hist = (uint32_t*) mem_alloc(3 * NR_TASKLETS * sizeof(uint32_t));
        memset(wr_hist, 0x0, 3 * NR_TASKLETS * sizeof(uint32_t));
    }
    barrier_wait(&index_barrier);

    uint32_t histogram_offset = 0;
    uint32_t histogram_size = 0;
    PIMKey_t* wram_keys = (PIMKey_t*) wr_buffer3[tasklet_id];
    PIMKey_t* mram_in_keys = (PIMKey_t*) (mram_heap + MRAM_INPUT_OFFSET);
    PIMKey_t* mram_out_keys = (PIMKey_t*) (mram_heap + MRAM_OUTPUT_OFFSET);

    /* histogram sizes */
    for (uint32_t i = tasklet_id * NR_KEYS_PER_WRAM_BUFFER;
                  i < (2 * dpu_args.num_keys);
                  i += NR_TASKLETS * NR_KEYS_PER_WRAM_BUFFER) {
        mram_read((__mram_ptr const void*) &mram_in_keys[i],
                                            wram_keys, WRAM_BUFFER_SIZE);
        uint32_t num_keys =
            ((i + NR_KEYS_PER_WRAM_BUFFER) < (2 * dpu_args.num_keys)) ?
            (NR_KEYS_PER_WRAM_BUFFER) :
            ((2 * dpu_args.num_keys) - i);

        for (uint32_t j = 0; j < num_keys; j += 2) {
            PIMKey_t key = wram_keys[j];
            uint32_t tmap = sdbm(key) % NR_TASKLETS;
            mutex_pool_lock(&partitioning_mtx, tmap);
            wr_hist[tmap]++;
            mutex_pool_unlock(&partitioning_mtx, tmap);
        }
    }
    barrier_wait(&index_barrier);

    /* histogram_size = wr_hist[tasklet_id]; */
    wr_hist[NR_TASKLETS + tasklet_id] = wr_hist[tasklet_id];
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

    /* histogram_offset = wr_hist[tasklet_id]; */
    wr_hist[2 * NR_TASKLETS + tasklet_id] = wr_hist[tasklet_id];
    barrier_wait(&index_barrier);

    /* tasklet mapping offsets */
    for (uint32_t i = tasklet_id * NR_KEYS_PER_WRAM_BUFFER;
                  i < (2 * dpu_args.num_keys);
                  i += NR_TASKLETS * NR_KEYS_PER_WRAM_BUFFER) {
        mram_read((__mram_ptr const void*) &mram_in_keys[i],
                                            wram_keys, WRAM_BUFFER_SIZE);
        uint32_t num_keys =
            ((i + NR_KEYS_PER_WRAM_BUFFER) < (2 * dpu_args.num_keys)) ?
            (NR_KEYS_PER_WRAM_BUFFER) :
            ((2 * dpu_args.num_keys) - i);

        for (uint32_t j = 0; j < num_keys; j += 2) {
            PIMKey_t key = wram_keys[j];
            uint32_t tmap = sdbm(key) % NR_TASKLETS;
            mutex_pool_lock(&partitioning_mtx, tmap);
            uint32_t mram_offset = wr_hist[tmap];
            wr_hist[tmap]++;
            mram_write(&wram_keys[j],
                (__mram_ptr void*) &mram_out_keys[2 * mram_offset], 2 * KEY_SIZE);
            mutex_pool_unlock(&partitioning_mtx, tmap);
        }
    }
    barrier_wait(&index_barrier);

    if (tasklet_id == 0) {
        if (wr_hist[NR_TASKLETS - 1] != dpu_args.num_keys) {
            dpu_args.kret = count_mismatch;
            exit(EXIT_FAILURE);
        }
    }
    barrier_wait(&index_barrier);

    /* insert */
    /* if (tasklet_id == 0) {
        for (uint32_t tid = 0; tid < NR_TASKLETS; tid++) { */
    for (uint32_t tid = tasklet_id; tid < NR_TASKLETS; tid += NR_TASKLETS) {
        uint32_t inserted_keys = 0;
        histogram_size = wr_hist[NR_TASKLETS + tid];
        histogram_offset = wr_hist[2 * NR_TASKLETS + tid];
        PIMKey_t* mram_keys = &mram_out_keys[2 * histogram_offset];

        for (uint32_t i = 0; i < (2 * histogram_size); i += NR_KEYS_PER_WRAM_BUFFER) {
            mram_read((__mram_ptr const void*) &mram_keys[i],
                                                wram_keys, WRAM_BUFFER_SIZE);
            uint32_t num_keys =
                ((i + NR_KEYS_PER_WRAM_BUFFER) < (2 * histogram_size)) ?
                (NR_KEYS_PER_WRAM_BUFFER) :
                ((2 * histogram_size) - i);

            for (uint32_t j = 0; j < num_keys; j += 2) {
                OpRet ret = insert(wram_keys[j], wram_keys[j + 1], tid);
                if (ret != entry_inserted) {
                    if (ret == duplicate_key) {
                        /* dpu_args.kret = not_unique; */
                    }
                    else {
                        /* dpu_args.kret = insert_failure; */
                    }
                    /* exit(EXIT_FAILURE); */
                }
                else {
                    inserted_keys++;
                }
            }
        }

        *((uint32_t*) wr_buffer2[tid]) = inserted_keys;
    }
    /* } */
    barrier_wait(&index_barrier);
    if (tasklet_id == 0) {
        uint32_t num_keys = 0;
        for (uint32_t i = 0; i < NR_TASKLETS; i++) {
            num_keys += *((uint32_t*) wr_buffer2[i]);
        }
        // printf("dpu_args.num_keys %u - %u\n", dpu_args.num_keys, num_keys);
        if (num_keys == dpu_args.num_keys) {
            dpu_args.kret = exec_success;
        }
        else {
            dpu_args.kret = count_mismatch;
        }
    }

    return 0;
}

#else

int insert_kernel() {

    uint32_t tasklet_id = me();

    wr_buffer2[tasklet_id] = mem_alloc(WRAM_BUFFER_SIZE);
    wr_buffer3[tasklet_id] = mem_alloc(WRAM_BUFFER_SIZE);

    uint32_t inserted_keys = 0;
    PIMKey_t* wram_keys = (PIMKey_t*) wr_buffer3[tasklet_id];
    PIMKey_t* mram_keys = (PIMKey_t*) (mram_heap + MRAM_INPUT_OFFSET);

    for (uint32_t i = 0; i < (2 * dpu_args.num_keys); i += NR_KEYS_PER_WRAM_BUFFER) {
        mram_read((__mram_ptr const void*) &mram_keys[i],
                                            wram_keys, WRAM_BUFFER_SIZE);
        uint32_t num_keys =
            ((i + NR_KEYS_PER_WRAM_BUFFER) < (2 * dpu_args.num_keys)) ?
            (NR_KEYS_PER_WRAM_BUFFER) :
            ((2 * dpu_args.num_keys) - i);

        for (uint32_t j = 0; j < num_keys; j += 2) {
            uint32_t tid = key_to_tasklet_hash(wram_keys[j]) % NR_TASKLETS;
            if (tid != tasklet_id) {
                continue;
            }
            OpRet ret = insert(wram_keys[j], wram_keys[j + 1], tasklet_id);
            if (ret != entry_inserted) {
                if (ret == duplicate_key) {
                    /* dpu_args.kret = not_unique; */
                }
                else {
                    /* dpu_args.kret = insert_failure; */
                }
                /* exit(EXIT_FAILURE); */
            }
            else {
                inserted_keys++;
            }
        }
    }

    // printf("Tasklet: %u | Inserts: %u\n", tasklet_id, inserted_keys);
    *((uint32_t*) wr_buffer2[tasklet_id]) = inserted_keys;
    barrier_wait(&index_barrier);
    if (tasklet_id == 0) {
        uint32_t num_keys = 0;
        for (uint32_t i = 0; i < NR_TASKLETS; i++) {
            num_keys += *((uint32_t*) wr_buffer2[i]);
        }
        // printf("dpu_args.num_keys %u - %u\n", dpu_args.num_keys, num_keys);
        if (num_keys == dpu_args.num_keys) {
            dpu_args.kret = exec_success;
        }
        else {
            dpu_args.kret = count_mismatch;
        }
    }

    return 0;
}
#endif

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

int (*kernels[3])() = {insert_kernel, logging_kernel, recovery_kernel};

int main() {
    return kernels[dpu_args.kernel]();
}
