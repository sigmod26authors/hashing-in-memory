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

MUTEX_INIT(coopr_flag_mtx);
MUTEX_INIT(coopr_wait_mtx);
MUTEX_INIT(coopr_active_mtx);
MUTEX_INIT(coopr_tasklets_mtx);
MUTEX_POOL_INIT(bucket_locks_mtx, 4);
BARRIER_INIT(index_barrier, NR_TASKLETS);

__host struct pimindex_dpu_args_t dpu_args;


/* MRAM pointers */
char* index_file_head;
char* index_file_head1;
char* index_file_head2;
char* chunks_offset;
IndexFile* index_file;
IndexFile* index_file1;
IndexFile* index_file2;
uint32_t* mptr1;
uint32_t* mptr2;
uint32_t** mptrs;

/* WRAM pointers */
void *wr_pool;
void **wr_buffer;
void **wr_buffer2;
void **wr_buffer3;
uint32_t* wr_chunk_bitmap;
uint8_t *wr_chunk_hdr_locks;
CBLock *wr_chunk_locks;
CBLock *wr_bucket_locks;
uint32_t* wr_hist;
PIMKey_t* wr_part_buffer;
uint8_t* wr_part_counter;
uint8_t* wr_coopr_wait;
#ifdef PINNED_CHUNK_HEADERS
ChunkHeader* wr_chunk_headers;
ChunkHeader* wr_chunk_headers1;
ChunkHeader* wr_chunk_headers2;
#endif

uint32_t prefix;
uint8_t global_depth;
uint32_t num_chunk_hdrs;
uint32_t nskips;
uint32_t coopr_flag;
uint32_t coopr_active;
uint32_t coopr_tasklets;
uint64_t next_free_chunk_id;
uint64_t index_file_lock; /* [63:32] unused, [31:] ptr, [30:] lock bit, [29:0] version */

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

static inline void* chunk_ptr(uint32_t cid) {
    return chunks_offset + cid * CHUNK_SIZE;
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

static inline uint32_t bucket_lock_pos(uint32_t cid, uint32_t bid) {
    return (cid * BUCKET_LOCKS_PER_CHUNK) + (bid & (BUCKET_LOCKS_PER_CHUNK - 1));
}

static inline bool try_acquire_bucket_lock(uint32_t pos, uint32_t tasklet_id) {
    mutex_pool_lock(&bucket_locks_mtx, pos);
    uint32_t* ptr = (uint32_t*) &wr_bucket_locks[pos];
    uint32_t old_lock_val = *ptr;
    mutex_pool_unlock(&bucket_locks_mtx, pos);

    CBLock old_lock = *((CBLock*) &old_lock_val);
    if ((old_lock.ref_count > 0) && (old_lock.tasklet_id != tasklet_id)) {
        return false;
    }

    CBLock new_lock = old_lock;
    new_lock.tasklet_id = tasklet_id;
    new_lock.ref_count++;
    uint32_t new_lock_val = *((uint32_t*) &new_lock);

    mutex_pool_lock(&bucket_locks_mtx, pos);
    ptr = (uint32_t*) &wr_bucket_locks[pos];
    if (*ptr != old_lock_val) {
        mutex_pool_unlock(&bucket_locks_mtx, pos);
        return false;
    }
    *ptr = new_lock_val;
    mutex_pool_unlock(&bucket_locks_mtx, pos);

    return true;
}

static inline void acquire_bucket_lock(uint32_t pos, uint32_t tasklet_id) {
    while (!try_acquire_bucket_lock(pos, tasklet_id));
}

static inline void release_bucket_lock(uint32_t pos) {
    mutex_pool_lock(&bucket_locks_mtx, pos);
    uint32_t* ptr = (uint32_t*) &wr_bucket_locks[pos];
    uint32_t lock_val = *ptr;
    mutex_pool_unlock(&bucket_locks_mtx, pos);

    CBLock lock = *((CBLock*) &lock_val);
    if (lock.ref_count > 0) {
        lock.ref_count--;
        lock.version++;

        uint32_t val = *((uint32_t*) &lock);
        mutex_pool_lock(&bucket_locks_mtx, pos);
        *ptr = val;
        mutex_pool_unlock(&bucket_locks_mtx, pos);
    }
}

void alloc_chunks(uint32_t* cids) {
    uint32_t cnt = 0;
    if (next_free_chunk_id > (MAX_NUM_CHUNKS - 1)) {
        printf("No more chunks...\n");
        exit(EXIT_FAILURE);
    }

    do {
        uint32_t old_free_cid = (uint32_t) next_free_chunk_id;
        uint32_t byte_pos = old_free_cid / 32u;
        uint32_t bit_pos, old_val, new_val;
        byte_pos--;

        do {
            byte_pos++;
            old_val = wr_chunk_bitmap[byte_pos];
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

        wr_chunk_bitmap[byte_pos] = new_val;
        for (uint32_t i = 0; i < bits; i++) {
            cids[cnt + i] += (byte_pos * 32);
        }
        cnt += bits;

        uint32_t last_free_slot = cids[cnt - 1];
        uint64_t val = (uint64_t)last_free_slot + 1;

        next_free_chunk_id = val;
        if ((next_free_chunk_id > (MAX_NUM_CHUNKS - 1)) && (cnt < 4)) {
            printf("No more chunks\n");
            exit(EXIT_FAILURE);
        }
    }
    while (cnt != 4);
}

void free_chunk(uint32_t cid) {
    uint32_t byte_pos = cid / 32;
    uint32_t bit_pos = cid % 32;

    uint32_t old_val, new_val;
    old_val = wr_chunk_bitmap[byte_pos];
    new_val = old_val | (1u << (31 - bit_pos));
    wr_chunk_bitmap[byte_pos] = new_val;

    uint32_t old_next_free_cid = (uint32_t) next_free_chunk_id;
    if (old_next_free_cid < cid) {
        return;
    }
    next_free_chunk_id = (uint64_t) cid;
}

#ifdef PINNED_CHUNK_HEADERS
void expand_index_file(uint32_t hdr_pos, ChunkHeader* new_chunk_headers,
                                                    uint32_t tasklet_id) {
    ChunkHeader* new_wram_chunk_headers;
    uint32_t lock = (uint32_t) index_file_lock;
    if (lock & 0x80000000u) {
        new_wram_chunk_headers = wr_chunk_headers1;
    }
    else {
        new_wram_chunk_headers = wr_chunk_headers2;
    }

    for (uint32_t i = 0; i < num_chunk_hdrs; i++) {
        uint32_t j = (i << 2);
        ChunkHeader* wram_new_hdrs = &new_wram_chunk_headers[j];
        for (uint32_t k = 0; k < 4; k++) {
            wram_new_hdrs[k].local_depth = wr_chunk_headers[i].local_depth;
            wram_new_hdrs[k].hash_scheme = wr_chunk_headers[i].hash_scheme;
            wram_new_hdrs[k].cid = wr_chunk_headers[i].cid;
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

    global_depth += 2;
    num_chunk_hdrs = (1u << global_depth);
    wr_chunk_headers = new_wram_chunk_headers;
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
            wr_chunk_headers[pos].cid = headers[i].cid;
            wr_chunk_headers[pos].hash_scheme = headers[i].hash_scheme;
            wr_chunk_headers[pos].local_depth = headers[i].local_depth;
        }
    }
}

#else

void expand_index_file(uint32_t hdr_pos, ChunkHeader* new_chunk_headers, uint32_t tasklet_id) {
    IndexFile* new_mram_index_file;
    IndexFile* old_mram_index_file = index_file;
    uint32_t lock = (uint32_t) index_file_lock;
    if (lock & 0x80000000u) {
        new_mram_index_file = (IndexFile*) index_file1;
        index_file_head = index_file_head1;
    }
    else {
        new_mram_index_file = (IndexFile*) index_file2;
        index_file_head = index_file_head2;
    }

    for (uint32_t i = 0; i < num_chunk_hdrs; i++) {
        ChunkHeader* mram_old_hdrs = &old_mram_index_file->chunk_headers[i];
        ChunkHeader* wram_old_hdrs = (ChunkHeader*) wr_buffer2[tasklet_id];
        mram_read((__mram_ptr const void*) mram_old_hdrs, wram_old_hdrs, sizeof(ChunkHeader));

        ChunkHeader* wram_new_hdrs = (ChunkHeader*) wr_buffer[tasklet_id];
        for (uint32_t k = 0; k < 4; k++) {
            wram_new_hdrs[k].local_depth = wram_old_hdrs->local_depth;
            wram_new_hdrs[k].hash_scheme = wram_old_hdrs->hash_scheme;
            wram_new_hdrs[k].cid = wram_old_hdrs->cid;
        }

        uint32_t j = (i << 2);
        ChunkHeader* mram_new_hdrs = &new_mram_index_file->chunk_headers[j];
        mram_write(wram_new_hdrs, (__mram_ptr void*) mram_new_hdrs, sizeof(ChunkHeader) * 4);
    }

    ChunkHeader* wram_new_hdrs = (ChunkHeader*) wr_buffer[tasklet_id];
    mram_read((__mram_ptr const void*) new_chunk_headers, wram_new_hdrs, sizeof(ChunkHeader) * 4);

    uint32_t jj = (hdr_pos << 2);
    ChunkHeader* mram_new_hdrs = &new_mram_index_file->chunk_headers[jj];
    mram_write(wram_new_hdrs, (__mram_ptr void*) mram_new_hdrs, sizeof(ChunkHeader) * 4);

    uint8_t* wram_index_file_head = (uint8_t*) wr_buffer[tasklet_id];
    wram_index_file_head[0] = global_depth + 2;
    mram_write(wram_index_file_head, (__mram_ptr void*) index_file_head, 8);

    global_depth += 2;
    num_chunk_hdrs = (1u << global_depth);
    index_file = new_mram_index_file;
}

void update_index_file(uint32_t hdr_pos, ChunkHeader* new_chunk_headers, uint8_t old_depth, uint32_t tasklet_id) {
    uint8_t depth_diff = global_depth - old_depth;
    uint32_t start_pos = hdr_pos >> depth_diff << depth_diff;
    uint32_t old_num_chunk_diff = 1u << depth_diff;
    uint32_t new_num_chunk_diff = old_num_chunk_diff >> 2;

    ChunkHeader* wram_old_chunk_hdrs = (ChunkHeader*) wr_buffer2[tasklet_id];
    ChunkHeader* wram_new_chunk_hdrs = (ChunkHeader*) wr_buffer[tasklet_id];
    mram_read((__mram_ptr const void*) new_chunk_headers, wram_new_chunk_hdrs, 4 * sizeof(ChunkHeader));

    uint32_t set_chunks = WRAM_BUFFER_SIZE / sizeof(ChunkHeader);
    for (uint32_t i = 0; i < 4; i++) {
        uint32_t pos = start_pos + new_num_chunk_diff * i;
        for (uint32_t j = pos; j < (pos + new_num_chunk_diff); j += set_chunks) {
            uint32_t chunks = ((j + set_chunks) < (pos + new_num_chunk_diff)) ? (set_chunks) : ((pos + new_num_chunk_diff) - j);

            for (uint32_t k = 0; k < chunks; k++) {
                wram_old_chunk_hdrs[k].cid = wram_new_chunk_hdrs[i].cid;
                wram_old_chunk_hdrs[k].hash_scheme = wram_new_chunk_hdrs[i].hash_scheme;
                wram_old_chunk_hdrs[k].local_depth = wram_new_chunk_hdrs[i].local_depth;
            }

            ChunkHeader* mram_old_chunk_hdrs = &index_file->chunk_headers[j];
            mram_write(wram_old_chunk_hdrs, (__mram_ptr void*) mram_old_chunk_hdrs, chunks * sizeof(ChunkHeader));
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
            hash_val = fnv32a(old_entry->key);
            bid = bucket_id(hash_val);
            new_cid = new_cids[chunk_hdr_ext_pos(hash_val, new_depth)];

            Bucket* new_chunk = (Bucket*) chunk_ptr(new_cid);
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

void allocate_chunks(uint8_t old_local_depth, uint32_t* new_cids,
                                                        uint32_t tasklet_id) {
    alloc_chunks(new_cids);
    ChunkHeader* new_hdrs = (ChunkHeader*) wr_buffer[tasklet_id];
    for (uint32_t i = 0; i < 4; i++) {
        new_hdrs[i].local_depth = old_local_depth + 2;
        new_hdrs[i].hash_scheme = 1;
        new_hdrs[i].cid = new_cids[i];
    }

#ifdef PINNED_CHUNK_HEADERS
    ChunkHeader* mram_new_hdrs =
        &index_file1->chunk_headers[512 + tasklet_id * 16]; /* TODO: modify */
#else
    ChunkHeader* mram_new_hdrs =
        &index_file->chunk_headers[512 + tasklet_id * 16]; /* TODO: modify */
#endif
    mram_write(new_hdrs, (__mram_ptr void*) mram_new_hdrs, sizeof(ChunkHeader) * 4);

    for (uint32_t i = 0; i < 4; i++) {
        uint32_t new_cid = new_cids[i];
        Bucket* new_chunk = (Bucket*) chunk_ptr(new_cid);
        Bucket* new_bucket = (Bucket*) wr_buffer[tasklet_id];

        for (uint32_t j = 0; j < BUCKETS_PER_CHUNK; j++) {
            new_bucket->header.bitmap = (uint16_t) 0xc000;
            mram_write(new_bucket, (__mram_ptr void*) &new_chunk[j], sizeof(Bucket));
        }
    }
}

bool rehash_buckets(uint32_t pos,
        uint32_t old_cid, uint8_t old_local_depth, uint32_t* new_cids,
        uint32_t cooperative_flag, uint32_t cooperative_id, uint32_t tasklet_id) {

    Bucket* old_chunk = (Bucket*) chunk_ptr(old_cid);
    for (uint32_t i = cooperative_id; i < BUCKETS_PER_CHUNK; i += coopr_tasklets) {
        Bucket* old_bucket = &old_chunk[i];
        if (!single_hash_reassign_bucket_entries(old_bucket,
                    old_local_depth + 2, new_cids, tasklet_id)) {
            return false; /* TODO */
        }
    }

    if (cooperative_flag == tasklet_id) {
        uint32_t i = 0;
        while (i < NR_TASKLETS) {
            while (!mutex_trylock(coopr_wait_mtx)) {
                ;
            }
            if (wr_coopr_wait[i] == 1) {
                i++;
            }
            mutex_unlock(coopr_wait_mtx);
        }

#ifdef PINNED_CHUNK_HEADERS
        ChunkHeader* new_chunk_headers =
            &index_file1->chunk_headers[512 + tasklet_id * 16]; /* TODO: modify */
#else
        ChunkHeader* new_chunk_headers =
            &index_file->chunk_headers[512 + tasklet_id * 16]; /* TODO: modify */
#endif
        if (old_local_depth < global_depth) {
            update_index_file(pos, new_chunk_headers, old_local_depth, tasklet_id);
        }
        else {
            expand_index_file(pos, new_chunk_headers, tasklet_id);
            uint32_t lck = (uint32_t) index_file_lock;
            if (lck & 0x80000000) {
                lck &= 0x7FFFFFFF;
            }
            else {
                lck |= 0x80000000;
            }
            index_file_lock = (uint64_t) lck;
        }
        free_chunk(old_cid);

        coopr_tasklets = 0;
        mutex_lock(coopr_active_mtx);
        coopr_active = 0;
        mutex_unlock(coopr_active_mtx);
    }
    else {
        mutex_lock(coopr_wait_mtx);
        wr_coopr_wait[tasklet_id] = 1;
        mutex_unlock(coopr_wait_mtx);
        uint32_t active = 1;
        while (active) {
            while (!mutex_trylock(coopr_active_mtx)) {
                ;
            }
            active = coopr_active;
            mutex_unlock(coopr_active_mtx);
        }
    }

    return true;
}

static inline bool key_exists(Bucket* bucket, PIMKey_t key, uint8_t fgprint) {
    BucketHeader* header = &bucket->header;

    uint8_t cmp[ENTRIES_PER_BUCKET_PIM] = {0};
    for (uint32_t i = 0; i < ENTRIES_PER_BUCKET_PIM; i++) {
        cmp[i] = (header->fingerprints[i] == fgprint) ? 1 : 0;
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

OpRet single_hash_insert(PIMKey_t key, PIMValue_t val, HashValue_t hash,
                                uint32_t old_cid, uint32_t tasklet_id) {

    uint32_t bid = bucket_id(hash);
    uint32_t buc_lock_pos = bucket_lock_pos(old_cid, bid);
    acquire_bucket_lock(buc_lock_pos, tasklet_id);

    Bucket* chunk = (Bucket*) chunk_ptr(old_cid);
    Bucket* mram_bucket = &chunk[bid];
    Bucket* wram_bucket = (Bucket*) wr_buffer[tasklet_id];
    mram_read((__mram_ptr void const*) mram_bucket, wram_bucket, BUCKET_SIZE);

    uint8_t fgprint = fingerprint(hash);
    if (key_exists(wram_bucket, key, fgprint)) {
        release_bucket_lock(buc_lock_pos);
        return duplicate_key;
    }

    uint32_t bit_cnt;
    uint32_t buc_bitmap = (uint32_t) wram_bucket->header.bitmap;
    __builtin_cao_rr(bit_cnt, buc_bitmap);

    if (bit_cnt == (BUCKET_HEADER_SKIP + ENTRIES_PER_BUCKET_PIM)) {
        release_bucket_lock(buc_lock_pos);
        return bucket_full;
    }
    else {
        insert_bucket_entry(wram_bucket, key, val, fgprint);
        mram_write(wram_bucket, (__mram_ptr void*) mram_bucket, BUCKET_SIZE);
        release_bucket_lock(buc_lock_pos);
        return entry_inserted;
    }
}


OpRet insert(PIMKey_t key, PIMValue_t val, uint32_t tasklet_id) {
    OpRet ret;
    uint32_t cpr_pos;
    uint32_t cpr_cid;
    uint32_t cpr_depth;
    HashValue_t hash_val = fnv32a(key);
CHECK_LOCKS:
    ;
    mutex_lock(coopr_flag_mtx);
    uint32_t cprflag = coopr_flag;
    mutex_unlock(coopr_flag_mtx);

    if (cprflag != NR_TASKLETS) {
        uint32_t new_cids[4];
        uint32_t cpr_flag = cprflag;

        mutex_lock(coopr_tasklets_mtx);
        uint32_t cpr_id = coopr_tasklets;
        coopr_tasklets++;
        mutex_unlock(coopr_tasklets_mtx);
        mutex_lock(coopr_wait_mtx);
        wr_coopr_wait[tasklet_id] = 1;
        mutex_unlock(coopr_wait_mtx);

        if (cprflag == tasklet_id) {
            uint32_t i = 0;
            while (i < NR_TASKLETS) {
                while (!mutex_trylock(coopr_wait_mtx)) {
                    ;
                }
                if (wr_coopr_wait[i] == 1) {
                    i++;
                }
                mutex_unlock(coopr_wait_mtx);
            }

            allocate_chunks(cpr_depth, new_cids, tasklet_id);
            for (i = 0; i < NR_TASKLETS; i++) {
                ((uint32_t*)wr_buffer[i])[0] = new_cids[0];
                ((uint32_t*)wr_buffer[i])[1] = new_cids[1];
                ((uint32_t*)wr_buffer[i])[2] = new_cids[2];
                ((uint32_t*)wr_buffer[i])[3] = new_cids[3];
                ((uint32_t*)wr_buffer[i])[4] = cpr_pos;
                ((uint32_t*)wr_buffer[i])[5] = cpr_cid;
                ((uint32_t*)wr_buffer[i])[6] = cpr_depth;
            }

            coopr_active = 1;
            mutex_lock(coopr_flag_mtx);
            coopr_flag = NR_TASKLETS;
            mutex_unlock(coopr_flag_mtx);
            rehash_buckets(cpr_pos, cpr_cid, cpr_depth,
                            new_cids, cpr_flag, cpr_id, tasklet_id);
        }
        else {
            while (cprflag != NR_TASKLETS) {
                while (!mutex_trylock(coopr_flag_mtx)) {
                    ;
                }
                cprflag = coopr_flag;
                mutex_unlock(coopr_flag_mtx);
            }
            mutex_lock(coopr_wait_mtx);
            wr_coopr_wait[tasklet_id] = 0;
            mutex_unlock(coopr_wait_mtx);

            new_cids[0] = ((uint32_t*)wr_buffer[tasklet_id])[0];
            new_cids[1] = ((uint32_t*)wr_buffer[tasklet_id])[1];
            new_cids[2] = ((uint32_t*)wr_buffer[tasklet_id])[2];
            new_cids[3] = ((uint32_t*)wr_buffer[tasklet_id])[3];
            cpr_pos = ((uint32_t*)wr_buffer[tasklet_id])[4];
            cpr_cid = ((uint32_t*)wr_buffer[tasklet_id])[5];
            cpr_depth = ((uint32_t*)wr_buffer[tasklet_id])[6];
            rehash_buckets(cpr_pos, cpr_cid, cpr_depth,
                            new_cids, cpr_flag, cpr_id, tasklet_id);
        }
        mutex_lock(coopr_wait_mtx);
        wr_coopr_wait[tasklet_id] = 0;
        mutex_unlock(coopr_wait_mtx);

        goto CHECK_LOCKS;
    }

    uint32_t pos = chunk_hdr_pos(hash_val, global_depth);
#ifdef PINNED_CHUNK_HEADERS
    uint32_t cid = wr_chunk_headers[pos].cid;
    uint8_t depth = wr_chunk_headers[pos].local_depth;
#else
    ChunkHeader* mram_header = &index_file->chunk_headers[pos];
    ChunkHeader* header = (ChunkHeader*)wr_buffer[tasklet_id];
    mram_read((__mram_ptr const void*) mram_header, header, sizeof(ChunkHeader));
    uint8_t depth = header->local_depth;
    uint32_t cid = header->cid;
#endif

    ret = single_hash_insert(key, val, hash_val, cid, tasklet_id);

    if (ret == bucket_full) {
        mutex_lock(coopr_flag_mtx);
        if (coopr_flag == NR_TASKLETS) {
            coopr_flag = tasklet_id;
            cpr_pos = pos;
            cpr_cid = cid;
            cpr_depth = depth;
        }
        mutex_unlock(coopr_flag_mtx);
        goto CHECK_LOCKS;
    }
    else {
        return ret;
    }
}

int insert_kernel() {

    uint32_t tasklet_id = me();

    wr_buffer2[tasklet_id] = mem_alloc(WRAM_BUFFER_SIZE);
    wr_buffer3[tasklet_id] = mem_alloc(WRAM_BUFFER_SIZE);

    uint32_t inserted_keys = 0;
    PIMKey_t* wram_keys = (PIMKey_t*) wr_buffer3[tasklet_id];
    PIMKey_t* mram_keys = (PIMKey_t*) (mram_heap + MRAM_INPUT_OFFSET);

    for (uint32_t i = tasklet_id * NR_KEYS_PER_WRAM_BUFFER;
                  i < (2 * dpu_args.num_keys);
                  i += NR_TASKLETS * NR_KEYS_PER_WRAM_BUFFER) {
        mram_read((__mram_ptr const void*) &mram_keys[i],
                                            wram_keys, WRAM_BUFFER_SIZE);
        uint32_t num_keys =
            ((i + NR_KEYS_PER_WRAM_BUFFER) < (2 * dpu_args.num_keys)) ?
            (NR_KEYS_PER_WRAM_BUFFER) :
            ((2 * dpu_args.num_keys) - i);

        for (uint32_t j = 0; j < num_keys; j += 2) {
            OpRet ret = insert(wram_keys[j], wram_keys[j + 1], tasklet_id);
            if (ret != entry_inserted) {
                if (ret == duplicate_key) {
                    dpu_args.kret = not_unique;
                }
                else {
                    dpu_args.kret = insert_failure;
                }
                exit(EXIT_FAILURE);
            }
            inserted_keys++;
        }
    }

    // printf("Tasklet: %u | Inserts: %u\n", tasklet_id, inserted_keys);
    *((uint32_t*) wr_buffer2[tasklet_id]) = inserted_keys;
    mutex_lock(coopr_wait_mtx);
    wr_coopr_wait[tasklet_id] = 1;
    mutex_unlock(coopr_wait_mtx);
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

int logging_kernel() {

    uint32_t tasklet_id = me();

    if (tasklet_id == 0) {
        uint32_t* wr_state = (uint32_t*)wr_buffer[tasklet_id];
        uint32_t* mr_state = (uint32_t*)((char*)mram_heap +
                                    MRAM_LOG_OFFSET + 64 * KiB);
        uint32_t i = 0;
        wr_state[i++] = (uint32_t)index_file_head;
        wr_state[i++] = (uint32_t)index_file_head1;
        wr_state[i++] = (uint32_t)index_file_head2;
        wr_state[i++] = (uint32_t)chunks_offset;
        wr_state[i++] = (uint32_t)index_file;
        wr_state[i++] = (uint32_t)index_file1;
        wr_state[i++] = (uint32_t)index_file2;
        wr_state[i++] = (uint32_t)mptr1;
        wr_state[i++] = (uint32_t)mptr2;
        /* TODO: mptrs */
        wr_state[i++] = prefix;
        wr_state[i++] = global_depth;
        wr_state[i++] = num_chunk_hdrs;
        wr_state[i++] = nskips;
        wr_state[i++] = next_free_chunk_id;
        wr_state[i++] = (index_file_lock >> 32);  /* TODO: msb == 0 */
        wr_state[i++] = (index_file_lock << 32) >> 32;
        /* actual size: 64 bytes */
        mram_write(wr_state, (__mram_ptr void*) mr_state, 64);

        mr_state = (uint32_t*)((char*)mr_state + (64 + 64));
        wr_state = wr_chunk_bitmap;
        /* actual size: BITMAP_LEN * sizeof(uint32_t) */
        mram_write(wr_state, (__mram_ptr void*) mr_state, 64);

        mr_state = (uint32_t*)((char*)mr_state + 64);
        wr_state = (uint32_t*)wr_chunk_hdr_locks;
        /* actual size: 512 * sizeof(uint8_t) */
        mram_write(wr_state, (__mram_ptr void*) mr_state, 512);

        mr_state = (uint32_t*)((char*)mr_state + (512 + 512));
        /* actual size: MAX_NUM_CHUNKS * sizeof(CBLock) */
        uint32_t locks_per_buffer = WRAM_BUFFER_SIZE / sizeof(CBLock);
        for (uint32_t i = 0; i < MAX_NUM_CHUNKS; i += locks_per_buffer) {
            mram_write(&wr_chunk_locks[i], (__mram_ptr void*) &mr_state[i],
                                                            WRAM_BUFFER_SIZE);
        }
        mr_state = (uint32_t*)((char*)mr_state + (1536 + 1536));
        /* actual size: NR_BUCKET_LOCKS * sizeof(CBLock) */
        for (uint32_t i = 0; i < NR_BUCKET_LOCKS; i += locks_per_buffer) {
            mram_write(&wr_bucket_locks[i], (__mram_ptr void*) &mr_state[i],
                                                            WRAM_BUFFER_SIZE);
        }

#ifdef PINNED_CHUNK_HEADERS
        if (dpu_args.log_level == 2) {
            mr_state = (uint32_t*)((char*)mr_state + (12288 + 1024));
            ChunkHeader* mr_chunk_headers = (ChunkHeader*) mr_state;
            /* actual size: BUCKET_LOCKS_PER_CHUNK * sizeof(CBLock) */
            uint32_t chunk_hdrs_per_buffer = WRAM_BUFFER_SIZE / sizeof(ChunkHeader);
            for (uint32_t i = 0; i < 512; i += chunk_hdrs_per_buffer) {
                mram_write(&wr_chunk_headers1[i],
                    (__mram_ptr void*) &mr_chunk_headers[i], WRAM_BUFFER_SIZE);
            }
            mr_state = (uint32_t*)((char*)mr_state + (4096 + 1024));
            mr_chunk_headers = (ChunkHeader*) mr_state;
            /* actual size: BUCKET_LOCKS_PER_CHUNK * sizeof(CBLock) */
            for (uint32_t i = 0; i < 512; i += chunk_hdrs_per_buffer) {
                mram_write(&wr_chunk_headers2[i],
                    (__mram_ptr void*) &mr_chunk_headers[i], WRAM_BUFFER_SIZE);
            }
        }
#endif
    }
    barrier_wait(&index_barrier);

    return 0;
}

int recovery_kernel() {

    uint32_t tasklet_id = me();

    if (tasklet_id == 0) {
        mem_reset();

        wr_buffer = (void**) mem_alloc(NR_TASKLETS * sizeof(void*));
        wr_buffer2 = (void**) mem_alloc(NR_TASKLETS * sizeof(void*));
        wr_buffer3 = (void**) mem_alloc(NR_TASKLETS * sizeof(void*));
        wr_chunk_bitmap = (uint32_t*) mem_alloc(BITMAP_LEN * sizeof(uint32_t));
        wr_chunk_hdr_locks = (uint8_t*) mem_alloc(512 * sizeof(uint8_t));
        /* TODO: update 512. use bits. memory/performance tradeoff */
        wr_chunk_locks = (CBLock*) mem_alloc(MAX_NUM_CHUNKS * sizeof(CBLock));
        wr_bucket_locks = (CBLock*) mem_alloc(NR_BUCKET_LOCKS * sizeof(CBLock));
#ifdef PINNED_CHUNK_HEADERS
        if (dpu_args.log_level == 2) {
            wr_chunk_headers1 = (ChunkHeader*) mem_alloc(512 * sizeof(ChunkHeader));
            wr_chunk_headers2 = (ChunkHeader*) mem_alloc(512 * sizeof(ChunkHeader));
        }
#endif
        coopr_flag = NR_TASKLETS;
        coopr_active = 0;
        coopr_tasklets = 0;
        wr_coopr_wait = (uint8_t*) mem_alloc(NR_TASKLETS * sizeof(uint8_t));
        memset(wr_coopr_wait, 0x0, NR_TASKLETS * sizeof(uint8_t));
    }
    barrier_wait(&index_barrier);

    wr_buffer[tasklet_id] = mem_alloc(WRAM_BUFFER_SIZE);

    if (tasklet_id == 0) {
        uint32_t* wr_state = (uint32_t*)wr_buffer[tasklet_id];
        uint32_t* mr_state = (uint32_t*)((char*)mram_heap +
                                    MRAM_LOG_OFFSET + 64 * KiB);
        mram_read((__mram_ptr const void*) mr_state, wr_state, 64);

        uint32_t i = 0;
        index_file_head = (char*)(wr_state[i++]);
        index_file_head1 = (char*)(wr_state[i++]);
        index_file_head2 = (char*)(wr_state[i++]);
        chunks_offset = (char*)(wr_state[i++]);
        index_file = (IndexFile*)(wr_state[i++]);
        index_file1 = (IndexFile*)(wr_state[i++]);
        index_file2 = (IndexFile*)(wr_state[i++]);
        mptr1 = (uint32_t*)(wr_state[i++]);
        mptr2 = (uint32_t*)(wr_state[i++]);
        /* TODO: mptrs */
        prefix = wr_state[i++];
        global_depth = (uint8_t)(wr_state[i++]);
        num_chunk_hdrs = wr_state[i++];
        nskips = wr_state[i++];
        next_free_chunk_id = (uint64_t)(wr_state[i++]);
        uint64_t msb = (uint64_t)(wr_state[i++]);
        uint64_t lsb = (uint64_t)(wr_state[i++]);
        msb = (msb << 32); /* TODO: msb == 0 */
        index_file_lock = msb | lsb;

        mr_state = (uint32_t*)((char*)mr_state + (64 + 64));
        wr_state = wr_chunk_bitmap;
        mram_read((__mram_ptr const void*) mr_state, wr_state, 64);

        mr_state = (uint32_t*)((char*)mr_state + 64);
        wr_state = (uint32_t*)wr_chunk_hdr_locks;
        mram_read((__mram_ptr const void*) mr_state, wr_state, 512);

        mr_state = (uint32_t*)((char*)mr_state + (512 + 512));
        uint32_t locks_per_buffer = WRAM_BUFFER_SIZE / sizeof(CBLock);
        for (uint32_t i = 0; i < MAX_NUM_CHUNKS; i += locks_per_buffer) {
            mram_read((__mram_ptr const void*) &mr_state[i],
                                        &wr_chunk_locks[i], WRAM_BUFFER_SIZE);
        }

        mr_state = (uint32_t*)((char*)mr_state + (1536 + 1536));
        for (uint32_t i = 0; i < NR_BUCKET_LOCKS; i += locks_per_buffer) {
            mram_read((__mram_ptr const void*) &mr_state[i],
                                        &wr_bucket_locks[i], WRAM_BUFFER_SIZE);
        }

#ifdef PINNED_CHUNK_HEADERS
        if (dpu_args.log_level == 2) {
            mr_state = (uint32_t*)((char*)mr_state + (12288 + 1024));
            ChunkHeader* mr_chunk_headers = (ChunkHeader*) mr_state;
            uint32_t chunk_hdrs_per_buffer = WRAM_BUFFER_SIZE / sizeof(ChunkHeader);
            for (uint32_t i = 0; i < 512; i += chunk_hdrs_per_buffer) {
                mram_read((__mram_ptr const void*) &mr_chunk_headers[i],
                                            &wr_chunk_headers1[i], WRAM_BUFFER_SIZE);
            }

            mr_state = (uint32_t*)((char*)mr_state + (4096 + 1024));
            mr_chunk_headers = (ChunkHeader*) mr_state;
            for (uint32_t i = 0; i < 512; i += chunk_hdrs_per_buffer) {
                mram_read((__mram_ptr const void*) &mr_chunk_headers[i],
                                            &wr_chunk_headers2[i], WRAM_BUFFER_SIZE);
            }

            wr_chunk_headers =
                (lsb & 0x80000000u) ? wr_chunk_headers2 : wr_chunk_headers1;
        }
#endif
    }
    barrier_wait(&index_barrier);

    return 0;
}

int (*kernels[3])() = {insert_kernel, logging_kernel, recovery_kernel};

int main() {
    return kernels[dpu_args.kernel]();
}
