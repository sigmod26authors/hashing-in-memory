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

MUTEX_INIT(index_file_lock_mtx);
MUTEX_POOL_INIT(bucket_locks_mtx, 4);
MUTEX_POOL_INIT(chunk_hdr_locks_mtx, 8);
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
#ifdef PINNED_CHUNK_HEADERS
ChunkHeader* wr_chunk_headers;
ChunkHeader* wr_chunk_headers1;
ChunkHeader* wr_chunk_headers2;
#endif

uint32_t prefix;
uint8_t global_depth;
uint32_t num_chunk_hdrs;
uint32_t nskips;
uint64_t next_free_chunk_id;
uint64_t index_file_lock; /* [63:32] unused, [31:] ptr, [30:] lock bit, [29:0] version */

const uint32_t index_file_lock_mask = (1u << 30);
const uint32_t mram_heap = (uint32_t)DPU_MRAM_HEAP_POINTER;


#ifdef HASH32
static inline uint32_t chunk_hdr_pos(HashValue_t hash, uint8_t mask_bits) {
    return (hash << 18) >> (32 - mask_bits);
}

static inline uint32_t bucket_id(HashValue_t hash) {
    return (hash << 8) >> 22;
}
#else
static inline uint32_t chunk_hdr_pos(HashValue_t hash, uint8_t mask_bits) {
    return (hash << 32) >> (64 - mask_bits);
}

static inline uint32_t bucket_id(HashValue_t hash) {
    return (hash << 8) >> 54;
}
#endif

static inline char* align_8B(char* ptr) {
    /* return ptr += ((8 - ((uint32_t)ptr % 8)) % 8); */
    uint32_t rem = (uint32_t)ptr % 8;
    return (rem == 0) ? (ptr) : (ptr + 8 - rem);
}

static inline char* align_bucket_size(char* ptr) {
    /* return ptr += ((BUCKET_SIZE - ((uint32_t)ptr % BUCKET_SIZE)) % BUCKET_SIZE); */
    uint32_t rem = (uint32_t)ptr % BUCKET_SIZE;
    return (rem == 0) ? (ptr) : (ptr + BUCKET_SIZE - rem);
}

static inline void* chunk_ptr(uint32_t cid) {
    return chunks_offset + cid * CHUNK_SIZE;
}

static inline uint32_t bucket_lock_pos(uint32_t cid, uint32_t bid) {
    return (cid * BUCKET_LOCKS_PER_CHUNK) + (bid & (BUCKET_LOCKS_PER_CHUNK - 1));
}

#ifdef INDEX_DELETE
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

#else

static inline bool try_acquire_chunk_hdr_lock(uint32_t pos) {
    mutex_pool_lock(&chunk_hdr_locks_mtx, pos);
    if (wr_chunk_hdr_locks[pos] == 1) {
        mutex_pool_unlock(&chunk_hdr_locks_mtx, pos);
        return false;
    }
    wr_chunk_hdr_locks[pos] = 1;
    mutex_pool_unlock(&chunk_hdr_locks_mtx, pos);
    return true;
}

static inline void acquire_chunk_hdr_lock(uint32_t pos) {
    while (!try_acquire_chunk_hdr_lock(pos));
}

static inline void release_chunk_hdr_lock(uint32_t pos) {
    mutex_pool_lock(&chunk_hdr_locks_mtx, pos);
    wr_chunk_hdr_locks[pos] = 0;
    mutex_pool_unlock(&chunk_hdr_locks_mtx, pos);
}


static inline uint32_t get_bucket_lock_val(uint32_t pos) {
    mutex_pool_lock(&bucket_locks_mtx, pos);
    uint32_t* ptr = (uint32_t*) &wr_bucket_locks[pos];
    uint32_t lock_val = *ptr;
    mutex_pool_unlock(&bucket_locks_mtx, pos);
    return lock_val;
}

static inline bool is_locked(uint32_t* lock_val) {
    CBLock lock = *((CBLock*)lock_val);
    return lock.ref_count > 0;
}
#endif

uint32_t alloc_chunk() {
    uint32_t old_free_cid = (uint32_t)next_free_chunk_id;
    uint32_t byte_pos = old_free_cid / 32u;
    uint32_t old_val, new_val, bit_pos;
    byte_pos--;

    do {
        byte_pos++;
        old_val = wr_chunk_bitmap[byte_pos];
    }
    while (old_val == 0);

    __builtin_clz_rr(bit_pos, old_val);
    new_val = old_val & ~(1u << (31 - bit_pos));
    wr_chunk_bitmap[byte_pos] = new_val;

    uint32_t free_slot = byte_pos * 32 + bit_pos;
    uint32_t val = (uint64_t)free_slot + 1;
    next_free_chunk_id = val;

    return free_slot;
}

void initialize_index(uint32_t tasklet_id) {

    if (tasklet_id == 0) {
        printf("Tasklet: %u\n", tasklet_id);
        mem_reset();

        wr_buffer = (void**) mem_alloc(NR_TASKLETS * sizeof(void*));
        wr_chunk_bitmap = (uint32_t*) mem_alloc(BITMAP_LEN * sizeof(uint32_t));
        wr_chunk_hdr_locks = (uint8_t*) mem_alloc(512 * sizeof(uint8_t));
         /* TODO: use bits. memory/performance tradeoff */
        wr_chunk_locks = (CBLock*) mem_alloc(MAX_NUM_CHUNKS * sizeof(CBLock));
        wr_bucket_locks = (CBLock*) mem_alloc(NR_BUCKET_LOCKS * sizeof(CBLock));
#ifdef PINNED_CHUNK_HEADERS
        wr_chunk_headers1 = (ChunkHeader*) mem_alloc(512 * sizeof(ChunkHeader));
        wr_chunk_headers2 = (ChunkHeader*) mem_alloc(512 * sizeof(ChunkHeader));
#endif
    }
    barrier_wait(&index_barrier);

    wr_buffer[tasklet_id] = mem_alloc(WRAM_BUFFER_SIZE);

    for (uint32_t i = tasklet_id; i < BITMAP_LEN; i += NR_TASKLETS) {
        wr_chunk_bitmap[i] = (uint32_t)(-1); /* set all bits */
    }
    barrier_wait(&index_barrier); /* TODO */

    for (uint32_t i = tasklet_id; i < 512; i += NR_TASKLETS) {
        wr_chunk_hdr_locks[i] = 0;
    }

    for (uint32_t i = tasklet_id; i < MAX_NUM_CHUNKS; i += NR_TASKLETS) {
        uint32_t* ptr = (uint32_t*) &wr_chunk_locks[i];
        *ptr = 0u;
    }

    for (uint32_t i = tasklet_id; i < NR_BUCKET_LOCKS; i += NR_TASKLETS) {
        uint32_t* ptr = (uint32_t*) &wr_bucket_locks[i];
        *ptr = 0u;
    }

    if (tasklet_id == 0) { /* TODO: multi-tasklet exec */

        num_chunk_hdrs = (1u << INIT_GLOBAL_DEPTH);
        wr_chunk_bitmap[0] = 0x7FFFFFFFu;
        next_free_chunk_id = 1;
        index_file_lock = 0;
        nskips = 0;

        mptr2 = (uint32_t*) (mram_heap + dpu_args.index_offs);
        char* ptr = (char*)mptr2 + 4;
        mptrs = (uint32_t**) mem_alloc(12 * sizeof(uint32_t*));

        uint32_t* wr_ids = (uint32_t*) wr_buffer[tasklet_id];

        for (uint32_t i = 0; i < 12; i++) {
            mptrs[i] = (uint32_t*) align_8B(ptr);

            wr_ids[0] = alloc_chunk();
            mram_write(wr_ids, (__mram_ptr void*) mptrs[i], 8);
             /* TODO */
            ptr = (char*)mptrs[i] + 32;
        }

        mptr1 = (uint32_t*) align_8B(ptr);
        ptr = (char*)mptr1 + 1024; /* TODO */

        /* index_file_head1 = align_chunk_size(ptr); */
        index_file_head1 = align_bucket_size(ptr);
        index_file1 = (IndexFile*) (index_file_head1 + 64); /* TODO */
        index_file_head2 = index_file_head1 + CHUNK_SIZE;
        index_file2 = (IndexFile*) (index_file_head2 + 64);
        chunks_offset = index_file_head2 + CHUNK_SIZE;

        index_file_head = index_file_head1;
        index_file = index_file1;

        for (uint32_t i = 0; i < num_chunk_hdrs; i++) {
            uint32_t cid = alloc_chunk();
            /* TODO: store ids in WRAM and update all headers first */
            Bucket* chunk = (Bucket*) chunk_ptr(cid);

            Bucket* wr_buc = (Bucket*) wr_buffer[tasklet_id];
            for (uint32_t j = 0; j < BUCKETS_PER_CHUNK; j++) {
                wr_buc->header.bitmap = (uint16_t) 0xC000;
                mram_write(wr_buc, (__mram_ptr void*) &chunk[j], sizeof(Bucket));
            }

#ifdef PINNED_CHUNK_HEADERS
            wr_chunk_headers1[i].local_depth = INIT_GLOBAL_DEPTH;
            wr_chunk_headers1[i].hash_scheme = 1;
            wr_chunk_headers1[i].cid = cid;
#else
            ChunkHeader* wr_hdr = (ChunkHeader*) wr_buffer[tasklet_id];
            ChunkHeader* mr_hdr = &index_file->chunk_headers[i];
            wr_hdr->local_depth = INIT_GLOBAL_DEPTH;
            wr_hdr->hash_scheme = 1;
            wr_hdr->cid = cid;
            mram_write(wr_hdr, (__mram_ptr void*) mr_hdr, sizeof(ChunkHeader));
#endif
        }

#ifdef PINNED_CHUNK_HEADERS
        wr_chunk_headers = wr_chunk_headers1;
        global_depth = INIT_GLOBAL_DEPTH;
#else
        uint8_t* wram_index_file_head = (uint8_t*) wr_buffer[tasklet_id];
        wram_index_file_head[0] = INIT_GLOBAL_DEPTH;
        global_depth = INIT_GLOBAL_DEPTH;
        mram_write(wram_index_file_head, (__mram_ptr void*) index_file_head, 64);
        /* TODO: merge with the write of the first chunk */
#endif

    }
    barrier_wait(&index_barrier);
}

#ifdef INDEX_DELETE
bool delete(PIMKey_t key, uint32_t tasklet_id) {
    HashValue_t hash_val = fnv32a(key);
    uint32_t pos = chunk_hdr_pos(hash_val, global_depth);

#ifdef PINNED_CHUNK_HEADERS
    uint32_t cid = wr_chunk_headers[pos].cid;
#else
    ChunkHeader* mram_hdr = &index_file->chunk_headers[pos];
    ChunkHeader* chdr = (ChunkHeader*)wr_buffer[tasklet_id];
    mram_read((__mram_ptr const void*) mram_hdr, chdr, sizeof(ChunkHeader));
    uint32_t cid = chdr->cid;
#endif

    uint32_t bid = bucket_id(hash_val);
    uint32_t lock_pos = bucket_lock_pos(cid, bid);
    acquire_bucket_lock(lock_pos, tasklet_id);

    Bucket* chunk = (Bucket*) chunk_ptr(cid);
    Bucket* mram_buc = &chunk[bid];
    Bucket* wram_buc = (Bucket*) wr_buffer[tasklet_id];

    mram_read((__mram_ptr void const*) mram_buc, wram_buc, BUCKET_SIZE);

    BucketHeader* bhdr = &wram_buc->header;

    uint8_t cmp[ENTRIES_PER_BUCKET_PIM] = {0};
    for (uint32_t i = 0; i < ENTRIES_PER_BUCKET_PIM; i++) {
        uint16_t bts = (wram_buc->header.bitmap &
            ((uint16_t)1u << (ENTRIES_PER_BUCKET_PIM - 1 - i)));
        cmp[i] = bts ? 1 : 0;
    }

    for (uint32_t i = 0; i < ENTRIES_PER_BUCKET_PIM; i++) {
        if (cmp[i]) {
            if (wram_buc->entries[BUCKET_HEADER_SKIP + i].key == key) {
                bhdr->bitmap = bhdr->bitmap & ~(1u << (13 - i));
                mram_write(wram_buc, (__mram_ptr void*) mram_buc, BUCKET_SIZE);
                /* mram_write(bhdr,
                    (__mram_ptr void*) &mram_buc->header, sizeof(BucketHeader)); */
                release_bucket_lock(lock_pos);
                return true;
            }
        }
    }

    release_bucket_lock(lock_pos);
    return false;
}

int delete_kernel() {

    uint32_t tasklet_id = me();

    wr_buffer2[tasklet_id] = mem_alloc(WRAM_BUFFER_SIZE);
    wr_buffer3[tasklet_id] = mem_alloc(WRAM_BUFFER_SIZE);

    uint32_t deleted = 0;
    uint32_t not_deleted = 0;
    PIMKey_t* wram_keys = (PIMKey_t*) wr_buffer3[tasklet_id];
    PIMKey_t* mram_keys = (PIMKey_t*) (mram_heap + MRAM_INPUT_OFFSET);

    for (uint32_t i = tasklet_id * NR_KEYS_PER_WRAM_BUFFER;
                  i < dpu_args.num_keys;
                  i += NR_TASKLETS * NR_KEYS_PER_WRAM_BUFFER) {
        mram_read((__mram_ptr const void*) &mram_keys[i],
                                            wram_keys, WRAM_BUFFER_SIZE);
        uint32_t num_keys = ((i + NR_KEYS_PER_WRAM_BUFFER) < dpu_args.num_keys) ?
                            (NR_KEYS_PER_WRAM_BUFFER) :
                            (dpu_args.num_keys - i);

        for (uint32_t j = 0; j < num_keys; j++) {
            if (wram_keys[j] == 0) {
                continue;
            }
            bool key_deleted = delete(wram_keys[j], tasklet_id);
            if (key_deleted) {
                deleted++;
            }
            else {
                not_deleted++;
            }
        }
    }
    // printf("Tasklet: %u | Deleted: %u\n", tasklet_id, deleted);
    
    *((uint32_t*) wr_buffer2[tasklet_id]) = deleted;
    *((uint32_t*) wr_buffer[tasklet_id]) = not_deleted;
    barrier_wait(&index_barrier);
    if (tasklet_id == 0) {
        uint32_t num_deleted = 0;
        uint32_t num_not_deleted = 0;
        for (uint32_t i = 0; i < NR_TASKLETS; i++) {
            num_deleted += *((uint32_t*) wr_buffer2[i]);
            num_not_deleted += *((uint32_t*) wr_buffer[i]);
        }
        if (num_not_deleted == 0) {
            // printf("Tasklet: %u | num_deleted: %u\n", tasklet_id, num_deleted);
            dpu_args.num_keys = num_deleted;
        }
        else {
            dpu_args.num_keys = 0;
        }
    }

    return 0;
}

#else

PIMValue_t search_bucket(PIMKey_t key, Bucket* bucket,
                    uint32_t lock_pos, uint32_t tasklet_id) {
    PIMValue_t ret;
    uint32_t lock_val;
CHECK_LOCKS:
    ;
    lock_val = get_bucket_lock_val(lock_pos);
    if (is_locked(&lock_val)) {
        goto CHECK_LOCKS;
    }

    Bucket* mram_bucket = bucket;
    Bucket* wram_bucket = (Bucket*) wr_buffer[tasklet_id];
    mram_read((__mram_ptr void const*) mram_bucket, wram_bucket, BUCKET_SIZE);

    for (uint32_t i = 0; i < ENTRIES_PER_BUCKET_PIM; i++) {
        if (wram_bucket->header.bitmap & (1u << (ENTRIES_PER_BUCKET_PIM - 1 - i))) {
            if (wram_bucket->entries[BUCKET_HEADER_SKIP + i].key == key) {
                ret = wram_bucket->entries[BUCKET_HEADER_SKIP + i].val;
                uint32_t lv = get_bucket_lock_val(lock_pos);
                if (lv != lock_val) {
                    goto CHECK_LOCKS;
                }
                return ret;
            }
        }
    }

    return (PIMValue_t)0;
}

PIMValue_t single_hash_search(PIMKey_t key, HashValue_t hash,
                                uint32_t cid, uint32_t tasklet_id) {
    uint32_t bid = bucket_id(hash);
    uint32_t lock_pos = bucket_lock_pos(cid, bid);
    Bucket* chunk = (Bucket*) chunk_ptr(cid);
    Bucket* bucket = &chunk[bid];

    return search_bucket(key, bucket, lock_pos, tasklet_id);
}

PIMValue_t search(PIMKey_t key, uint32_t tasklet_id) {
    PIMValue_t ret;
    HashValue_t hash_val = fnv32a(key);
CHECK_LOCKS:
    ;
    mutex_lock(index_file_lock_mtx);
    uint32_t old_index_file_lock = (uint32_t)index_file_lock;
    mutex_unlock(index_file_lock_mtx);
    if (old_index_file_lock & index_file_lock_mask) {
        goto CHECK_LOCKS;
    }

    uint32_t pos = chunk_hdr_pos(hash_val, global_depth);
#ifdef PINNED_CHUNK_HEADERS
    acquire_chunk_hdr_lock(pos);
    uint32_t cid = wr_chunk_headers[pos].cid;
    release_chunk_hdr_lock(pos);
#else
    ChunkHeader* mram_hdr = &index_file->chunk_headers[pos];
    ChunkHeader* hdr = (ChunkHeader*)wr_buffer[tasklet_id];
    acquire_chunk_hdr_lock(pos);
    mram_read((__mram_ptr const void*) mram_hdr, hdr, sizeof(ChunkHeader));
    release_chunk_hdr_lock(pos);
    uint32_t cid = hdr->cid;
#endif

    ret = single_hash_search(key, hash_val, cid, tasklet_id);

    mutex_lock(index_file_lock_mtx);
    uint32_t lock = (uint32_t)index_file_lock;
    mutex_unlock(index_file_lock_mtx);
    if (lock != old_index_file_lock) {
        goto CHECK_LOCKS;
    }

    return ret;
}

int search_kernel() {

    uint32_t tasklet_id = me();

    wr_buffer2[tasklet_id] = mem_alloc(WRAM_BUFFER_SIZE);
    wr_buffer3[tasklet_id] = mem_alloc(WRAM_BUFFER_SIZE);

    uint32_t found = 0;
    uint32_t not_found = 0;
    PIMKey_t* wram_keys = (PIMKey_t*) wr_buffer3[tasklet_id];
    PIMKey_t* mram_keys = (PIMKey_t*) (mram_heap + MRAM_INPUT_OFFSET);

    for (uint32_t i = tasklet_id * NR_KEYS_PER_WRAM_BUFFER;
                  i < dpu_args.num_keys;
                  i += NR_TASKLETS * NR_KEYS_PER_WRAM_BUFFER) {
        mram_read((__mram_ptr const void*) &mram_keys[i],
                                            wram_keys, WRAM_BUFFER_SIZE);
        uint32_t num_keys = ((i + NR_KEYS_PER_WRAM_BUFFER) < dpu_args.num_keys) ?
                            (NR_KEYS_PER_WRAM_BUFFER) :
                            (dpu_args.num_keys - i);

        for (uint32_t j = 0; j < num_keys; j++) {
            if (wram_keys[j] == 0) {
                continue;
            }
            PIMValue_t val = search(wram_keys[j], tasklet_id);
            if (val == DEFAULT_VALUE) {
                found++;
            }
            else {
                not_found++;
            }

            wram_keys[j] = val;
        }
#ifndef INDEX_NSEARCH
        mram_write(wram_keys, (__mram_ptr void*) &mram_keys[i], WRAM_BUFFER_SIZE);
#endif
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
#ifdef INDEX_NSEARCH
        if (num_found == 0) {
            dpu_args.num_keys = num_not_found;
        }
        else {
            dpu_args.num_keys = 0;
        }
#else
        if (num_not_found == 0) {
            dpu_args.kret = exec_success;
        }
        else {
            dpu_args.kret = count_mismatch;
        }
#endif
    }

    return 0;
}
#endif

int initialization_kernel() {
    uint32_t tasklet_id = me();
    initialize_index(tasklet_id);
    return 0;
}

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
            mutex_pool_lock(&chunk_hdr_locks_mtx, part);
            /* borrowing the mutex pool */
            wr_hist[part]++;
            mutex_pool_unlock(&chunk_hdr_locks_mtx, part);
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
            mutex_pool_lock(&chunk_hdr_locks_mtx, part);
            if (wr_part_counter[part] == 0) {
                wr_part_buffer[2 * part] = key;
                wr_part_counter[part] = 1;
                mutex_pool_unlock(&chunk_hdr_locks_mtx, part);
            }
            else {
                wr_part_buffer[2 * part + 1] = key;
                mr_offs = wr_hist[part];
                wr_hist[part] += 2;
                wr_part_counter[part] = 0;
                mutex_pool_unlock(&chunk_hdr_locks_mtx, part);
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
    PIMKey_t* mr_outkeys =
        (PIMKey_t*)((char*)mram_heap + MRAM_OUTPUT_OFFSET);
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
            mutex_pool_lock(&chunk_hdr_locks_mtx, part);
            /* borrowing the mutex pool */
            wr_hist[part]++;
            mutex_pool_unlock(&chunk_hdr_locks_mtx, part);
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
            mutex_pool_lock(&chunk_hdr_locks_mtx, part);
            uint32_t mr_offs = wr_hist[part];
            wr_hist[part]++;
            mram_write(&wr_keys[j],
                    (__mram_ptr void*) &mr_outkeys[2 * mr_offs], 2 * KEY_SIZE);
            mutex_pool_unlock(&chunk_hdr_locks_mtx, part);
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

    if (tasklet_id == 0) {
        uint32_t alloced_chunks = 0;
        uint32_t unalloced_chunks = 0;
        uint32_t chunks_cnt = 0;
        uint32_t buckets_cnt = 0;
        uint32_t keys_cnt = 0;
        uint32_t prev_cid = (uint32_t)(-1);

        for (uint32_t i = 0; i < BITMAP_LEN; i++) {
            uint32_t byte = wr_chunk_bitmap[i];
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
        /*printf("Allocated chunks: %u (+13)\n", (alloced_chunks - 13));
        printf("Unallocated chunks: %u\n", unalloced_chunks);
        printf("Chunk headers: %u\n", num_chunk_hdrs);
        printf("Next free chunk id: %lu\n", next_free_chunk_id);*/

#ifdef PINNED_CHUNK_HEADERS
        uint32_t* wr_state = (uint32_t*)wr_buffer[tasklet_id];
        uint32_t* mr_state = (uint32_t*)((char*)mram_heap +
                                    MRAM_LOG_OFFSET + 64 * KiB);
        mram_read((__mram_ptr const void*) mr_state, wr_state, 64);
        uint32_t lsb = wr_state[15];
        /* logging space for chunk headers */
        mr_state = (uint32_t*)((char*)mr_state + (64 + 64) + 64 + (512 + 512) +
                                                 (1536 + 1536) + (12288 + 1024));
        mr_state = (lsb & 0x80000000u) ? 
            (uint32_t*)((char*)mr_state + (4096 + 1024)) : mr_state;
#endif

        ChunkHeader hdr;
        ChunkHeader* wr_chunk_hdrs = &hdr;
        /* ChunkHeader* wr_chunk_hdrs = (ChunkHeader*) wr_buffer2[tasklet_id]; */
        Bucket* wr_buckets = (Bucket*) wr_buffer[tasklet_id];
        /* uint32_t chnk_hdrs_per_xfer = WRAM_BUFFER_SIZE / sizeof(ChunkHeader); */
        uint32_t chnk_hdrs_per_xfer = 1;
        uint32_t buckets_per_xfer = WRAM_BUFFER_SIZE / BUCKET_SIZE;

        for (uint32_t i = 0; i < num_chunk_hdrs; i += chnk_hdrs_per_xfer) {
            uint32_t headers = ((i + chnk_hdrs_per_xfer) < num_chunk_hdrs) ?
                               (chnk_hdrs_per_xfer) :
                               (num_chunk_hdrs - i);

#ifdef PINNED_CHUNK_HEADERS
            ChunkHeader* mr_chunk_hdrs = &((ChunkHeader*) mr_state)[i];
#else
            ChunkHeader* mr_chunk_hdrs = &index_file->chunk_headers[i];
#endif
            /* mram_read((__mram_ptr const void*) mr_chunk_hdrs, wr_chunk_hdrs, WRAM_BUFFER_SIZE); */
            mram_read((__mram_ptr const void*) mr_chunk_hdrs, wr_chunk_hdrs, sizeof(ChunkHeader));

            for (uint32_t j = 0; j < headers; j++) {
                uint32_t cid = wr_chunk_hdrs[j].cid;
                if (cid == prev_cid) {
                    continue;
                }

                Bucket* chunk = (Bucket*) chunk_ptr(cid);
                for (uint32_t k = 0; k < BUCKETS_PER_CHUNK; k += buckets_per_xfer) {
                    mram_read((__mram_ptr const void*) &chunk[k],
                                            wr_buckets, WRAM_BUFFER_SIZE);
                    uint32_t buckets =
                        ((k + buckets_per_xfer) < BUCKETS_PER_CHUNK) ?
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
        dpu_args.num_chunks = chunks_cnt;
        dpu_args.num_buckets = buckets_cnt;
        dpu_args.num_keys = keys_cnt;
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
        if (dpu_args.log_level == 77) {
            //
        }
        else {
            wr_bucket_locks = (CBLock*) mem_alloc(NR_BUCKET_LOCKS * sizeof(CBLock));
        }
#ifdef PINNED_CHUNK_HEADERS
        if (dpu_args.log_level == 2) {
            wr_chunk_headers1 = (ChunkHeader*) mem_alloc(512 * sizeof(ChunkHeader));
            wr_chunk_headers2 = (ChunkHeader*) mem_alloc(512 * sizeof(ChunkHeader));
        }
#endif
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

        if (dpu_args.log_level == 77) {
            /* TODO */
        }
        else {
            mr_state = (uint32_t*)((char*)mr_state + (1536 + 1536));
            for (uint32_t i = 0; i < NR_BUCKET_LOCKS; i += locks_per_buffer) {
                mram_read((__mram_ptr const void*) &mr_state[i],
                &wr_bucket_locks[i], WRAM_BUFFER_SIZE);
            }
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

int (*kernels[NR_KERNELS])() =
    {initialization_kernel, kvmapping_kernel, mapping_kernel,
#ifdef INDEX_DELETE
     delete_kernel,
#else
     search_kernel,
#endif
    mem_utilization_kernel, logging_kernel, recovery_kernel};

int main() {
    return kernels[dpu_args.kernel]();
}
