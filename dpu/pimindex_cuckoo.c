#include <defs.h>
#include <mram.h>
#include <alloc.h>
#include <barrier.h>
#include <stdint.h>
#include <stdio.h>
#include <mutex.h>
#include <mutex_pool.h>
#include <vmutex.h>
#include <string.h>
#include <stdlib.h>

#include "pimindex.h"
#include "hash.h"

MUTEX_INIT(index_lock_mtx);
MUTEX_INIT(lock_wait_mtx);
MUTEX_INIT(index_exponent_mtx);
#ifdef MULTITASKLET_EXPAND
MUTEX_INIT(expanders_mtx);
MUTEX_INIT(expand_index_done_mtx);
#endif
#ifdef MUTEX_POOL
MUTEX_POOL_INIT(bucket_locks_mtx, 32);
#elif defined(VIRTUAL_MUTEX)
VMUTEX_INIT(bucket_locks_vmtx, NR_BUCKET_LOCKS, 32);
#endif
BARRIER_INIT(index_barrier, NR_TASKLETS);

__host struct pimindex_dpu_args_t dpu_args;


/* MRAM pointers */
Bucket* buckets;
Bucket* buckets2;

/* WRAM pointers */
void *wr_pool;
void **wr_buffer;
void **wr_buffer2; /* TODO: single/per-tasklet */
void **wr_buffer3;
BucketLock *wr_bucket_locks;
uint32_t* wr_hist;
PIMKey_t* wr_part_buffer;
uint8_t* wr_part_counter;
uint8_t* wr_lock_wait;

uint32_t prefix;
uint32_t index_exponent;
uint32_t index_exponent2;
uint32_t index_lock;
#ifdef MULTITASKLET_EXPAND
uint32_t expanders;
uint32_t expand_index_done;
#endif

const uint32_t mram_heap = (uint32_t)DPU_MRAM_HEAP_POINTER;
const uint32_t num_bucket_locks = NR_BUCKET_LOCKS;
const uint32_t buckets_per_buffer = WRAM_BUFFER_SIZE / BUCKET_SIZE;


static inline void enqueue(CuckooSlotQueue* q, uint32_t bid,
                           uint8_t pathcode, int8_t hops) {
#ifdef KERNEL_ASSERT
    assert(q->last != MAX_CUCKOO_COUNT);
#endif
    q->slots[q->last].bid = bid;
    q->slots[q->last].pathcode = pathcode;
    q->slots[q->last].hops = hops;
    q->last++;
}

static inline void dequeue(CuckooSlotQueue* q, CuckooSlot* s) {
#ifdef KERNEL_ASSERT
    assert(q->first < q->last);
#endif
    s->bid = q->slots[q->first].bid;
    s->pathcode = q->slots[q->first].pathcode;
    s->hops = q->slots[q->first].hops;
    q->first++;
}

static inline void swap_pos(uint32_t* a, uint32_t* b) {
    if (*b < *a) {
        uint32_t tmp = *a;
        *a = *b;
        *b = tmp;
    }
}

static inline char* align_bucket_size(char* ptr) {
    /* return ptr += ((BUCKET_SIZE - ((uint32_t)ptr % BUCKET_SIZE)) % BUCKET_SIZE); */
    uint32_t rem = (uint32_t)ptr % BUCKET_SIZE;
    return (rem == 0) ? (ptr) : (ptr + BUCKET_SIZE - rem);
}

static inline uint8_t fingerprint(uint64_t hash) {
    const uint64_t hash_64bit = hash;
    const uint32_t hash_32bit = ((uint32_t)hash_64bit) ^ ((uint32_t)(hash_64bit >> 32));
    const uint32_t hash_16bit = ((uint16_t)hash_32bit) ^ ((uint16_t)(hash_32bit >> 16));
    const uint32_t hash_8bit = ((uint8_t)hash_16bit) ^ ((uint8_t)(hash_16bit >> 8));
    return hash_8bit;
}

static inline uint32_t bucket_id1(uint32_t hash, uint32_t mask) {
    return ((uint32_t)hash & mask);
}

static inline uint32_t bucket_id2(uint32_t bid1, uint8_t fgprint, uint32_t mask) {
    const uint64_t nz_tag = (uint64_t)fgprint + 1;
    const uint64_t val = ((uint64_t)bid1) ^ (nz_tag * 0xc6a4a7935bd1e995);
    return ((uint32_t)val & mask);
}

static inline uint32_t lock_pos(uint32_t bid) {
    return bid & (num_bucket_locks - 1);
}

static inline void set_bucket_lock_rehashed(uint32_t pos) {
#ifdef MUTEX_POOL
    mutex_pool_lock(&bucket_locks_mtx, pos);
    wr_bucket_locks[pos].rehashed = 1;
    mutex_pool_unlock(&bucket_locks_mtx, pos);
#elif defined(VIRTUAL_MUTEX)
    vmutex_lock(&bucket_locks_vmtx, pos);
    wr_bucket_locks[pos].rehashed = 1;
    vmutex_unlock(&bucket_locks_vmtx, pos);
#endif
}

static inline bool bucket_lock_rehashed(uint32_t pos) {
#ifdef MUTEX_POOL
    mutex_pool_lock(&bucket_locks_mtx, pos);
    if (wr_bucket_locks[pos].rehashed) {
        mutex_pool_unlock(&bucket_locks_mtx, pos);
        return true;
    }
    mutex_pool_unlock(&bucket_locks_mtx, pos);
    return false;
#elif defined(VIRTUAL_MUTEX)
    vmutex_lock(&bucket_locks_vmtx, pos);
    if (wr_bucket_locks[pos].rehashed) {
        vmutex_unlock(&bucket_locks_vmtx, pos);
        return true;
    }
    vmutex_unlock(&bucket_locks_vmtx, pos);
    return false;
#endif
}

static inline void increment_lock_key_count(uint32_t pos) {
#ifdef MUTEX_POOL
    mutex_pool_lock(&bucket_locks_mtx, pos);
    wr_bucket_locks[pos].key_count++;
    mutex_pool_unlock(&bucket_locks_mtx, pos);
#elif defined(VIRTUAL_MUTEX)
    vmutex_lock(&bucket_locks_vmtx, pos);
    wr_bucket_locks[pos].key_count++;
    vmutex_unlock(&bucket_locks_vmtx, pos);
#endif
}

static inline bool try_acquire_bucket_lock(uint32_t pos) {
#ifdef MUTEX_POOL
    mutex_pool_lock(&bucket_locks_mtx, pos);
    if (wr_bucket_locks[pos].locked) {
        mutex_pool_unlock(&bucket_locks_mtx, pos);
        return false;
    }
    wr_bucket_locks[pos].locked = 1;
    mutex_pool_unlock(&bucket_locks_mtx, pos);
    return true;
#elif defined(VIRTUAL_MUTEX)
    vmutex_lock(&bucket_locks_vmtx, pos);
    if (wr_bucket_locks[pos].locked) {
        vmutex_unlock(&bucket_locks_vmtx, pos);
        return false;
    }
    wr_bucket_locks[pos].locked = 1;
    vmutex_unlock(&bucket_locks_vmtx, pos);
    return true;
#endif
}

static inline void acquire_bucket_lock(uint32_t pos) {
    while (!try_acquire_bucket_lock(pos));
}

static inline void release_bucket_lock(uint32_t pos) {
#ifdef MUTEX_POOL
    mutex_pool_lock(&bucket_locks_mtx, pos);
    wr_bucket_locks[pos].locked = 0;
    mutex_pool_unlock(&bucket_locks_mtx, pos);
#elif defined(VIRTUAL_MUTEX)
    vmutex_lock(&bucket_locks_vmtx, pos);
    wr_bucket_locks[pos].locked = 0;
    vmutex_unlock(&bucket_locks_vmtx, pos);
#endif
}

void insert_bucket_entry(PIMKey_t key, PIMKey_t val, uint8_t fgprint,
                         BucketSlot* buc_slot, uint32_t tasklet_id) {
    Bucket* wr_bucket = (Bucket*) wr_buffer[tasklet_id];
    mram_read((__mram_ptr const void*) &buckets[buc_slot->bid],
                                        wr_bucket, BUCKET_SIZE);
    wr_bucket[0].entries[BUCKET_HEADER_SKIP + buc_slot->slot].key = key;
    wr_bucket[0].entries[BUCKET_HEADER_SKIP + buc_slot->slot].val = val;
    wr_bucket[0].header.fingerprints[BUCKET_HEADER_SKIP + buc_slot->slot] = fgprint;
    wr_bucket[0].header.bitmap[BUCKET_HEADER_SKIP + buc_slot->slot] = 1;
    mram_write(wr_bucket, (__mram_ptr void*) &buckets[buc_slot->bid], BUCKET_SIZE);
    increment_lock_key_count(lock_pos(buc_slot->bid));
}

bool key_exists(PIMKey_t key, uint32_t bid1, uint32_t bid2,
                BucketSlot* buc_slot, uint32_t tasklet_id) {
    Bucket* wr_bucket = (Bucket*) wr_buffer[tasklet_id];
    mram_read((__mram_ptr const void*) &buckets[bid1], wr_bucket, BUCKET_SIZE);
    for (uint32_t s = 0; s < ENTRIES_PER_BUCKET_PIM; s++) {
        if ((wr_bucket->header.bitmap[BUCKET_HEADER_SKIP + s] == 1) &&
                (wr_bucket->entries[BUCKET_HEADER_SKIP + s].key == key)) {
            buc_slot->bid = bid1;
            buc_slot->slot = s;
            return true;
        }
    }
    mram_read((__mram_ptr const void*) &buckets[bid2], wr_bucket, BUCKET_SIZE);
    for (uint32_t s = 0; s < ENTRIES_PER_BUCKET_PIM; s++) {
        if ((wr_bucket->header.bitmap[BUCKET_HEADER_SKIP + s] == 1) &&
                (wr_bucket->entries[BUCKET_HEADER_SKIP + s].key == key)) {
            buc_slot->bid = bid2;
            buc_slot->slot = s;
            return true;
        }
    }
    return false;
}

PIMKey_t find_key(PIMKey_t key, uint32_t bid1, uint32_t bid2, uint32_t tasklet_id) {
    Bucket* wr_bucket = (Bucket*) wr_buffer[tasklet_id];
    mram_read((__mram_ptr const void*) &buckets[bid1], wr_bucket, BUCKET_SIZE);
    for (uint32_t s = 0; s < ENTRIES_PER_BUCKET_PIM; s++) {
        if ((wr_bucket->header.bitmap[BUCKET_HEADER_SKIP + s] == 1) &&
                (wr_bucket->entries[BUCKET_HEADER_SKIP + s].key == key)) {
            return wr_bucket->entries[BUCKET_HEADER_SKIP + s].val;
        }
    }
    mram_read((__mram_ptr const void*) &buckets[bid2], wr_bucket, BUCKET_SIZE);
    for (uint32_t s = 0; s < ENTRIES_PER_BUCKET_PIM; s++) {
        if ((wr_bucket->header.bitmap[BUCKET_HEADER_SKIP + s] == 1) &&
                (wr_bucket->entries[BUCKET_HEADER_SKIP + s].key == key)) {
            return wr_bucket->entries[BUCKET_HEADER_SKIP + s].val;
        }
    }
    return (PIMValue_t)0;
}

void rehash_bucket(uint32_t old_exp, uint32_t new_exp,
                    uint32_t old_bid, uint32_t tasklet_id) {

    uint32_t new_slot = 0;
    uint32_t new_bid = old_bid + (1u << old_exp);
    Bucket* wr_bucket1 = (Bucket*) wr_buffer[tasklet_id];
    Bucket* wr_bucket2 = (Bucket*) wr_buffer2[tasklet_id];
    mram_read((__mram_ptr const void*) &buckets2[old_bid], wr_bucket2, BUCKET_SIZE);

    for (uint32_t old_slot = 0; old_slot < ENTRIES_PER_BUCKET_PIM; old_slot++) {
        if (!wr_bucket2->header.bitmap[BUCKET_HEADER_SKIP + old_slot]) {
            continue;
        }

        uint32_t bid;
        uint32_t slot;
        PIMKey_t key = wr_bucket2->entries[BUCKET_HEADER_SKIP + old_slot].key;
        uint64_t hash_val = murmur64(key);
        uint8_t fgprint = fingerprint(hash_val);
        uint32_t old_mask = (1u << old_exp) - 1;
        uint32_t old_bid1 = bucket_id1(hash_val, old_mask);
        uint32_t old_bid2 = bucket_id2(old_bid1, fgprint, old_mask);
        uint32_t new_mask = (1u << new_exp) - 1;
        uint32_t new_bid1 = bucket_id1(hash_val, new_mask);
        uint32_t new_bid2 = bucket_id2(new_bid1, fgprint, new_mask);

        if ((old_bid == old_bid1 && new_bid == new_bid1) ||
                (old_bid == old_bid2 && new_bid == new_bid2)) {
            bid = new_bid;
            slot = new_slot++;
        }
        else {
#ifdef KERNEL_ASSERT
            assert((old_bid == old_bid1 && new_bid1 == old_bid1) ||
                   (old_bid == old_bid2 && new_bid2 == old_bid2));
#endif
            bid = old_bid;
            slot = old_slot;
        }

        mram_read((__mram_ptr const void*) &buckets[bid], wr_bucket1, BUCKET_SIZE);
        wr_bucket1->entries[BUCKET_HEADER_SKIP + slot].key = key;
        wr_bucket1->entries[BUCKET_HEADER_SKIP + slot].val =
            wr_bucket2->entries[BUCKET_HEADER_SKIP + old_slot].val;
        wr_bucket1->header.fingerprints[BUCKET_HEADER_SKIP + slot] =
            wr_bucket2->header.fingerprints[BUCKET_HEADER_SKIP + old_slot];
        wr_bucket1->header.bitmap[BUCKET_HEADER_SKIP + slot] = 1;
        wr_bucket2->header.bitmap[BUCKET_HEADER_SKIP + old_slot] = 0;
        mram_write(wr_bucket1, (__mram_ptr void*) &buckets[bid], BUCKET_SIZE);
    }

    mram_write(wr_bucket2, (__mram_ptr void*) &buckets2[old_bid], BUCKET_SIZE);
}

void rehash_lock_buckets(uint32_t pos, uint32_t tasklet_id) {
    if (bucket_lock_rehashed(pos)) {
        return;
    }

    mutex_lock(index_exponent_mtx);
    uint32_t new_exp = index_exponent;
    uint32_t old_exp = index_exponent2;
    mutex_unlock(index_exponent_mtx);
#ifdef KERNEL_ASSERT
    assert(new_exp == (old_exp + 1));
#endif
    uint32_t num_old_buckets = (1u << old_exp);
    for (uint32_t i = pos; i < num_old_buckets; i += num_bucket_locks) {
        rehash_bucket(old_exp, new_exp, i, tasklet_id);
    }
    set_bucket_lock_rehashed(pos);
}

void initialize_index(uint32_t tasklet_id) {

    if (tasklet_id == 0) {
        printf("Tasklet: %u\n", tasklet_id);
        mem_reset();

        wr_pool = (void*) mem_alloc(4 /* max. buffer types */ * 16 /* max. tasklets */ * WRAM_BUFFER_SIZE);
        wr_buffer = (void**) mem_alloc(NR_TASKLETS * sizeof(void*));
        wr_buffer2 = (void**) mem_alloc(NR_TASKLETS * sizeof(void*));
        wr_buffer3 = (void**) mem_alloc(NR_TASKLETS * sizeof(void*));
        wr_bucket_locks = (BucketLock*) mem_alloc(12 * KiB);
        wr_lock_wait = (uint8_t*) mem_alloc(NR_TASKLETS * sizeof(uint8_t));
    }
    barrier_wait(&index_barrier);

    wr_buffer[tasklet_id] = (char*)wr_pool + tasklet_id * WRAM_BUFFER_SIZE;
    wr_buffer2[tasklet_id] = (char*)wr_pool + (NR_TASKLETS + tasklet_id) * WRAM_BUFFER_SIZE;
    wr_buffer3[tasklet_id] = (char*)wr_pool + (2 * NR_TASKLETS + tasklet_id) * WRAM_BUFFER_SIZE;

    if (tasklet_id == 0) { /* TODO: multi-tasklet exec */
        index_exponent = INIT_INDEX_EXPONENT;
        index_exponent2 = 0;
        index_lock = NR_TASKLETS;
#ifdef MULTITASKLET_EXPAND
        expanders = 0;
        expand_index_done = 1;
#endif

        char* ptr = (char*) (mram_heap + dpu_args.index_offs);
        buckets = (Bucket*) align_bucket_size(ptr);
        ptr = (char*) (mram_heap + dpu_args.index_offs + MAX_INDEX_SPACE);
        buckets2 = (Bucket*) align_bucket_size(ptr);
    }
    barrier_wait(&index_barrier);

    wr_lock_wait[tasklet_id] = 0;
    for (uint32_t i = tasklet_id; i < NR_BUCKET_LOCKS; i += NR_TASKLETS) {
        uint32_t* ptr = (uint32_t*) &wr_bucket_locks[i];
        *ptr = 0u;
#ifdef LAZY_REHASHING
        wr_bucket_locks[i].rehashed = 1;
#endif
    }

    uint32_t num_buckets = (1u << index_exponent);
    Bucket* wr_zeroed_buc = (Bucket*) wr_buffer[tasklet_id];
    memset(wr_zeroed_buc->header.bitmap, 0, 8);
    for (uint32_t i = tasklet_id; i < num_buckets; i += NR_TASKLETS) {
        mram_write(wr_zeroed_buc, (__mram_ptr void*) &buckets[i], BUCKET_SIZE);
    }
    for (uint32_t i = tasklet_id; i < num_buckets; i += NR_TASKLETS) {
        mram_write(wr_zeroed_buc, (__mram_ptr void*) &buckets2[i], BUCKET_SIZE);
    }
}

#ifdef MULTITASKLET_EXPAND
bool expand_index(uint32_t exp, bool coord, uint32_t expid, uint32_t tasklet_id) {
    uint32_t new_exp = exp + 1;
    if (new_exp > MAX_INDEX_EXPONENT) {
        printf("maximum index exponent exceeded...\n");
        exit(EXIT_FAILURE);
    }

#ifdef LAZY_REHASHING
    for (uint32_t i = expid; i < num_bucket_locks; i += expanders) {
        if (wr_bucket_locks[i].rehashed == 1) {
            wr_bucket_locks[i].rehashed = 0;
            continue;
        }
        uint32_t num_old_buckets = (1u << index_exponent2);
        for (uint32_t j = i; j < num_old_buckets; j += num_bucket_locks) {
            rehash_bucket(index_exponent2, index_exponent, j, tasklet_id);
        }
    }

    if (coord) {
        uint32_t tid = 0;
        while (tid < NR_TASKLETS) {
            while (!mutex_trylock(lock_wait_mtx)) {
                ;
            }
            if (wr_lock_wait[tid] == 1) {
                tid++;
            }
            mutex_unlock(lock_wait_mtx);
        }
        Bucket* ptr = buckets;
        buckets = buckets2;
        buckets2 = ptr;
        index_exponent2 = index_exponent;
        index_exponent = new_exp;
        expanders = 0;
        mutex_lock(expand_index_done_mtx);
        expand_index_done = 1;
        mutex_unlock(expand_index_done_mtx);
    }
    else {
        mutex_lock(lock_wait_mtx);
        wr_lock_wait[tasklet_id] = 1;
        mutex_unlock(lock_wait_mtx);
        uint32_t done = 0;
        while (!done) {
            while (!mutex_trylock(expand_index_done_mtx)) {
                ;
            }
            done = expand_index_done;
            mutex_unlock(expand_index_done_mtx);
        }
    }
#else
    for (uint32_t i = expid; i < num_bucket_locks; i += expanders) {
        uint32_t num_old_buckets = (1u << index_exponent2);
        for (uint32_t j = i; j < num_old_buckets; j += num_bucket_locks) {
            rehash_bucket(index_exponent2, index_exponent, j, tasklet_id);
        }
    }

    if (coord) {
        uint32_t tid = 0;
        while (tid < NR_TASKLETS) {
            while (!mutex_trylock(lock_wait_mtx)) {
                ;
            }
            if (wr_lock_wait[tid] == 1) {
                tid++;
            }
            mutex_unlock(lock_wait_mtx);
        }
        expanders = 0;
        mutex_lock(expand_index_done_mtx);
        expand_index_done = 1;
        mutex_unlock(expand_index_done_mtx);
    }
    else {
        mutex_lock(lock_wait_mtx);
        wr_lock_wait[tasklet_id] = 1;
        mutex_unlock(lock_wait_mtx);
        uint32_t done = 0;
        while (!done) {
            while (!mutex_trylock(expand_index_done_mtx)) {
                ;
            }
            done = expand_index_done;
            mutex_unlock(expand_index_done_mtx);
        }
    }
#endif

    return true;
}

#else /* MULTITASKLET_EXPAND */

bool expand_index(uint32_t exp, uint32_t tasklet_id) {
    uint32_t new_exp = exp + 1;
    if (new_exp > MAX_INDEX_EXPONENT) {
        printf("maximum index exponent exceeded...\n");
        exit(EXIT_FAILURE);
    }

#ifdef KERNEL_ASSERT
    assert(index_exponent == exp);
    if (index_exponent2 != 0) {
        assert(index_exponent == (index_exponent2 + 1));
    }
#endif

#ifdef LAZY_REHASHING
    for (uint32_t i = 0; i < num_bucket_locks; i++) {
        if (wr_bucket_locks[i].rehashed == 1) {
            continue;
        }
        uint32_t num_old_buckets = (1u << index_exponent2);
        for (uint32_t j = i; j < num_old_buckets; j += num_bucket_locks) {
            rehash_bucket(index_exponent2, index_exponent, j, tasklet_id);
        }
    }

    Bucket* ptr = buckets;
    buckets = buckets2;
    buckets2 = ptr;
    index_exponent2 = index_exponent;
    index_exponent = new_exp;
    for (uint32_t i = 0; i < num_bucket_locks; i++) {
        wr_bucket_locks[i].rehashed = 0;
    }
#else
    Bucket* ptr = buckets;
    buckets = buckets2;
    buckets2 = ptr;
    index_exponent2 = index_exponent;
    index_exponent = new_exp;

    for (uint32_t i = 0; i < num_bucket_locks; i++) {
        uint32_t num_old_buckets = (1u << index_exponent2);
        for (uint32_t j = i; j < num_old_buckets; j += num_bucket_locks) {
            rehash_bucket(index_exponent2, index_exponent, j, tasklet_id);
        }
    }
#endif

    return true;
}
#endif /* MULTITASKLET_EXPAND */

void cuckoo_search_slot(uint32_t bid1, uint32_t bid2, uint32_t exp,
                        uint32_t mask, CuckooSlot* cuckoo_slot,
                        BucketSlot* buc_slot, uint32_t tasklet_id) {
    CuckooSlotQueue cuckoo_queue;
    cuckoo_queue.first = 0;
    cuckoo_queue.last = 0;
    enqueue(&cuckoo_queue, bid1, 0, 0);
    enqueue(&cuckoo_queue, bid2, 1, 0);

    while (cuckoo_queue.first != cuckoo_queue.last) {
        dequeue(&cuckoo_queue, cuckoo_slot);

        uint32_t lpos = lock_pos(cuckoo_slot->bid);
        acquire_bucket_lock(lpos);
        mutex_lock(index_exponent_mtx);
        uint32_t curr_exp = index_exponent;
        mutex_unlock(index_exponent_mtx);
        if (curr_exp != exp) {
            release_bucket_lock(lpos);
            buc_slot->ret = expand_conflict;
            return;
        }
#ifdef LAZY_REHASHING
        rehash_lock_buckets(lpos, tasklet_id);
#endif

        Bucket* wr_buc = (Bucket*) wr_buffer[tasklet_id];
        mram_read((__mram_ptr const void*) &buckets[cuckoo_slot->bid],
                                                    wr_buc, BUCKET_SIZE);

        uint32_t start = cuckoo_slot->pathcode % ENTRIES_PER_BUCKET_PIM;
        for (uint32_t i = 0; i < ENTRIES_PER_BUCKET_PIM; i++) {
            uint8_t slot = (start + i) % ENTRIES_PER_BUCKET_PIM;
            if (wr_buc[0].header.bitmap[BUCKET_HEADER_SKIP + slot] == 0) {
                /* empty slot found */
                cuckoo_slot->pathcode =
                            cuckoo_slot->pathcode * ENTRIES_PER_BUCKET_PIM + slot;
                release_bucket_lock(lpos);
                return;
            }

            if (cuckoo_slot->hops < (MAX_HOPS - 1)) {
#ifdef KERNEL_ASSERT
                assert(cuckoo_queue.last != MAX_CUCKOO_COUNT);
#endif
                uint8_t fgprint =
                        wr_buc[0].header.fingerprints[BUCKET_HEADER_SKIP + slot];
                enqueue(&cuckoo_queue,
                        (bucket_id2(cuckoo_slot->bid, fgprint, mask)),
                        (cuckoo_slot->pathcode * ENTRIES_PER_BUCKET_PIM + slot),
                        (cuckoo_slot->hops + 1));
            }
        }
        release_bucket_lock(lpos);
    }
    /* no empty slot found */
    cuckoo_slot->hops = -1;
}

int8_t cuckoo_search_path(uint32_t bid1, uint32_t bid2,
                          uint32_t exp, CuckooEntry* cuckoo_path,
                          BucketSlot* buc_slot, uint32_t tasklet_id) {
    CuckooSlot cuckoo_slot;
    uint32_t mask = (1u << exp) - 1;
    cuckoo_search_slot(bid1, bid2, exp, mask, &cuckoo_slot, buc_slot, tasklet_id);

    if (buc_slot->ret == expand_conflict) {
        return -1;
    }
    if (cuckoo_slot.hops == -1) {
        buc_slot->ret = index_full;
        return -1;
    }

    for (int i = cuckoo_slot.hops; i >= 0; i--) {
        cuckoo_path[i].slot = cuckoo_slot.pathcode % ENTRIES_PER_BUCKET_PIM;
        cuckoo_slot.pathcode /= ENTRIES_PER_BUCKET_PIM;
    }

    CuckooEntry* prev_cuckoo_entry;
    CuckooEntry* curr_cuckoo_entry = &cuckoo_path[0];
    if (cuckoo_slot.pathcode == 0) {
        curr_cuckoo_entry->bid = bid1;
    }
    else {
#ifdef KERNEL_ASSERT
        assert(cuckoo_slot.pathcode == 1);
#endif
        curr_cuckoo_entry->bid = bid2;
    }

    uint32_t lpos = lock_pos(curr_cuckoo_entry->bid);
    acquire_bucket_lock(lpos);
    mutex_lock(index_exponent_mtx);
    uint32_t curr_exp = index_exponent;
    mutex_unlock(index_exponent_mtx);
    if (curr_exp != exp) {
        release_bucket_lock(lpos);
        buc_slot->ret = expand_conflict;
        return -1;
    }
#ifdef LAZY_REHASHING
    rehash_lock_buckets(lpos, tasklet_id);
#endif

    Bucket* wr_bucket = (Bucket*) wr_buffer[tasklet_id];
    mram_read((__mram_ptr const void*) &buckets[curr_cuckoo_entry->bid],
                                                wr_bucket, BUCKET_SIZE);
    bool slot_empty =
        wr_bucket[0].header.bitmap[BUCKET_HEADER_SKIP + curr_cuckoo_entry->slot] == 0;
    if (slot_empty) {
        return 0;
    }

    PIMKey_t key =
        wr_bucket[0].entries[BUCKET_HEADER_SKIP + curr_cuckoo_entry->slot].key;
    release_bucket_lock(lpos);
    curr_cuckoo_entry->hash = murmur64(key);
    curr_cuckoo_entry->fgprint = fingerprint(curr_cuckoo_entry->hash);
    for (int8_t i = 1; i <= cuckoo_slot.hops; i++) {
        prev_cuckoo_entry = &cuckoo_path[i - 1];
#ifdef KERNEL_ASSERT
        uint32_t prev_bid1 = bucket_id1(prev_cuckoo_entry->hash, mask);
        uint32_t prev_bid2 = bucket_id2(prev_bid1, prev_cuckoo_entry->fgprint, mask);
        assert(prev_cuckoo_entry->bid == prev_bid1 ||
               prev_cuckoo_entry->bid == prev_bid2);
#endif

        /* next bucket on cuckoo path */
        curr_cuckoo_entry = &cuckoo_path[i];
        curr_cuckoo_entry->bid =
            bucket_id2(prev_cuckoo_entry->bid, prev_cuckoo_entry->fgprint, mask);

        lpos = lock_pos(curr_cuckoo_entry->bid);
        acquire_bucket_lock(lpos);
        mutex_lock(index_exponent_mtx);
        uint32_t curr_exp = index_exponent;
        mutex_unlock(index_exponent_mtx);
        if (curr_exp != exp) {
            release_bucket_lock(lpos);
            buc_slot->ret = expand_conflict;
            return -1;
        }
#ifdef LAZY_REHASHING
        rehash_lock_buckets(lpos, tasklet_id);
#endif

        mram_read((__mram_ptr const void*) &buckets[curr_cuckoo_entry->bid],
                                                    wr_bucket, BUCKET_SIZE);
        slot_empty = wr_bucket[0].header.bitmap[BUCKET_HEADER_SKIP +
                                                curr_cuckoo_entry->slot] == 0;
        if (slot_empty) {
            release_bucket_lock(lpos);
            return i;
        }

        key = wr_bucket[0].entries[BUCKET_HEADER_SKIP + curr_cuckoo_entry->slot].key;
        release_bucket_lock(lpos);
        curr_cuckoo_entry->hash = murmur64(key);
        curr_cuckoo_entry->fgprint = fingerprint(curr_cuckoo_entry->hash);
    }

    return cuckoo_slot.hops; /* TODO: check */
}

bool cuckoo_move_path(uint32_t bid1, uint32_t bid2, uint32_t exp,
                          int8_t hops, CuckooEntry* cuckoo_path,
                          BucketSlot* buc_slot, uint32_t tasklet_id) {
    CuckooEntry* cuckoo_entry = &cuckoo_path[0];
    Bucket* wr_buc = (Bucket*) wr_buffer[tasklet_id];

    if (hops == 0) {
        uint32_t bid = cuckoo_entry->bid;
        uint32_t slot = cuckoo_entry->slot;
#ifdef KERNEL_ASSERT
        /* assert(bid == bid1 || bid == bid2); */
#endif

        uint32_t lpos1 = lock_pos(bid1);
        uint32_t lpos2 = lock_pos(bid2);
        swap_pos(&lpos1, &lpos2);

        acquire_bucket_lock(lpos1);
        mutex_lock(index_exponent_mtx);
        uint32_t curr_exp = index_exponent;
        mutex_unlock(index_exponent_mtx);
        if (curr_exp != exp) {
            release_bucket_lock(lpos1);
            buc_slot->ret = expand_conflict;
            return false;
        }
        if (lpos1 != lpos2) {
            acquire_bucket_lock(lpos2);
        }
#ifdef LAZY_REHASHING
        rehash_lock_buckets(lpos1, tasklet_id);
        rehash_lock_buckets(lpos2, tasklet_id);
#endif

        mram_read((__mram_ptr const void*) &buckets[bid], wr_buc, BUCKET_SIZE);
        if (!wr_buc->header.bitmap[BUCKET_HEADER_SKIP + slot]) {
            /* keep the initial two buckets locked */
            return true;
        }
        else {
            release_bucket_lock(lpos1);
            if (lpos1 != lpos2) {
                release_bucket_lock(lpos2);
            }
            return false;
        }
    }

    CuckooEntry* src_cuckoo_entry = &cuckoo_path[hops - 1];
    CuckooEntry* dest_cuckoo_entry = &cuckoo_path[hops];
    uint32_t src_bid = src_cuckoo_entry->bid;
    uint32_t src_slot = src_cuckoo_entry->slot;
    uint32_t dest_bid = dest_cuckoo_entry->bid;
    uint32_t dest_slot = dest_cuckoo_entry->slot;
    Bucket* wr_src_buc = (Bucket*) wr_buffer[tasklet_id];
    Bucket* wr_dest_buc = (Bucket*) wr_buffer2[tasklet_id];

    uint32_t lpos1 = lock_pos(bid1);
    uint32_t lpos2 = lock_pos(bid2);
    uint32_t lpos3 = lock_pos(dest_bid);

    if (hops == 1) {
        swap_pos(&lpos2, &lpos3);
        swap_pos(&lpos1, &lpos3);
        swap_pos(&lpos1, &lpos2);

        acquire_bucket_lock(lpos1);
        mutex_lock(index_exponent_mtx);
        uint32_t curr_exp = index_exponent;
        mutex_unlock(index_exponent_mtx);
        if (curr_exp != exp) {
            release_bucket_lock(lpos1);
            buc_slot->ret = expand_conflict;
            return false;
        }
        if (lpos1 != lpos2) {
            acquire_bucket_lock(lpos2);
        }
        if (lpos2 != lpos3) {
            acquire_bucket_lock(lpos3);
        }
#ifdef LAZY_REHASHING
        rehash_lock_buckets(lpos1, tasklet_id);
        rehash_lock_buckets(lpos2, tasklet_id);
        rehash_lock_buckets(lpos3, tasklet_id);
#endif
    }
    else {
        printf("(hops > 1) currently not supported\n");
        exit(EXIT_FAILURE);
    }

    mram_read((__mram_ptr const void*) &buckets[src_bid], wr_src_buc, BUCKET_SIZE);
    mram_read((__mram_ptr const void*) &buckets[dest_bid], wr_dest_buc, BUCKET_SIZE);
    bool src_empty = wr_src_buc->header.bitmap[BUCKET_HEADER_SKIP + src_slot] == 0;
    bool dest_empty = wr_dest_buc->header.bitmap[BUCKET_HEADER_SKIP + dest_slot] == 0;
    PIMKey_t src_key = wr_src_buc->entries[BUCKET_HEADER_SKIP + src_slot].key;
    if (src_empty || !dest_empty || murmur64(src_key) != src_cuckoo_entry->hash) {
        release_bucket_lock(lpos1);
        if (lpos1 != lpos2) {
            release_bucket_lock(lpos2);
        }
        if (lpos2 != lpos3) {
            release_bucket_lock(lpos3);
        }
        return false;
    }

    wr_dest_buc->entries[BUCKET_HEADER_SKIP + dest_slot].key = src_key;
    wr_dest_buc->entries[BUCKET_HEADER_SKIP + dest_slot].val =
                wr_src_buc->entries[BUCKET_HEADER_SKIP + src_slot].val;
    wr_dest_buc->header.fingerprints[BUCKET_HEADER_SKIP + dest_slot] =
                wr_src_buc->header.fingerprints[BUCKET_HEADER_SKIP + src_slot];
    wr_dest_buc->header.bitmap[BUCKET_HEADER_SKIP + dest_slot] = 1;
    wr_src_buc->header.bitmap[BUCKET_HEADER_SKIP + src_slot] = 0;

    mram_write(wr_src_buc, (__mram_ptr void*) &buckets[src_bid], BUCKET_SIZE);
    mram_write(wr_dest_buc, (__mram_ptr void*) &buckets[dest_bid], BUCKET_SIZE);

    /* keep the initial two buckets locked */
    if (lock_pos(dest_bid) != lock_pos(bid1) &&
            lock_pos(dest_bid) != lock_pos(bid2)) {
        release_bucket_lock(lock_pos(dest_bid));
    }
    return true;
}

void cuckoo_find_bucket_slot(uint32_t bid1, uint32_t bid2, uint32_t exp,
                             BucketSlot* buc_slot, uint32_t tasklet_id) {
    uint32_t lpos1 = lock_pos(bid1);
    uint32_t lpos2 = lock_pos(bid2);
    release_bucket_lock(lpos1);
    if (lpos1 != lpos2) {
        release_bucket_lock(lpos2);
    }

    CuckooEntry cuckoo_path[MAX_HOPS];
SEARCH_PATH:
    ;
    int8_t hops =
        cuckoo_search_path(bid1, bid2, exp, cuckoo_path, buc_slot, tasklet_id);
    if (buc_slot->ret == expand_conflict || buc_slot->ret == index_full) {
        return;
    }

    bool moved =
        cuckoo_move_path(bid1, bid2, exp, hops, cuckoo_path, buc_slot, tasklet_id);
    if (buc_slot->ret == expand_conflict) {
        return;
    }
    else if (!moved) {
        goto SEARCH_PATH;
    }
    else {
        buc_slot->bid = cuckoo_path[0].bid;
        buc_slot->slot = cuckoo_path[0].slot;
#ifdef KERNEL_ASSERT
        assert(buc_slot->bid == bid1 || buc_slot->bid == bid2);
#endif
        buc_slot->ret = slot_found;
        return;
    }
}

bool try_find_slot(PIMKey_t key, uint32_t bid, uint32_t* slot, uint32_t tasklet_id) {
    *slot = (uint32_t)(-1);
    Bucket* wr_bucket = (Bucket*) wr_buffer[tasklet_id];
    mram_read((__mram_ptr const void*) &buckets[bid], wr_bucket, BUCKET_SIZE);
    for (uint32_t s = 0; s < ENTRIES_PER_BUCKET_PIM; s++) {
        if (wr_bucket->header.bitmap[BUCKET_HEADER_SKIP + s]) {
            if (wr_bucket->entries[BUCKET_HEADER_SKIP + s].key == key) {
                *slot = s;
                return false;
            }
        }
        else {
            *slot = s;
        }
    }
    return true;
}

void find_bucket_slot(PIMKey_t key, uint32_t bid1, uint32_t bid2,
                      uint32_t exp, BucketSlot* buc_slot, uint32_t tasklet_id) {
    uint32_t slot1;
    uint32_t slot2;
    if (!try_find_slot(key, bid1, &slot1, tasklet_id)) {
        buc_slot->bid = bid1;
        buc_slot->slot = slot1;
        buc_slot->ret = duplicate_key;
        return;
    }
    if (!try_find_slot(key, bid2, &slot2, tasklet_id)) {
        buc_slot->bid = bid2;
        buc_slot->slot = slot2;
        buc_slot->ret = duplicate_key;
        return;
    }
    if (slot1 != ((uint32_t)(-1))) {
        buc_slot->bid = bid1;
        buc_slot->slot = slot1;
        buc_slot->ret = slot_found;
        return;
    }
    if (slot2 != ((uint32_t)(-1))) {
        buc_slot->bid = bid2;
        buc_slot->slot = slot2;
        buc_slot->ret = slot_found;
        return;
    }

    /* cuckoo */
    cuckoo_find_bucket_slot(bid1, bid2, exp, buc_slot, tasklet_id);

    if (buc_slot->ret == slot_found) {
        /* check in case another insertion inserted the same key
           when we released the locks in cuckoo_find_bucket_slot */
        if (key_exists(key, bid1, bid2, buc_slot, tasklet_id)) {
            buc_slot->ret = duplicate_key;
        }
    }
}

#ifdef MULTITASKLET_EXPAND
OpRet insert(PIMKey_t key, PIMKey_t val, uint32_t tasklet_id) {
    BucketSlot buc_slot;
    uint64_t hash_val = murmur64(key);
    uint8_t fgprint = fingerprint(hash_val);

GET_LOCKS:
    ;
    mutex_lock(index_exponent_mtx);
    uint32_t exp = index_exponent;
    mutex_unlock(index_exponent_mtx);

    mutex_lock(index_lock_mtx);
    uint32_t idx_lock = index_lock;
    mutex_unlock(index_lock_mtx);
    if (idx_lock != NR_TASKLETS) {
        mutex_lock(expanders_mtx);
        uint32_t expid = expanders;
        expanders++;
        mutex_unlock(expanders_mtx);
        mutex_lock(lock_wait_mtx);
        wr_lock_wait[tasklet_id] = 1;
        mutex_unlock(lock_wait_mtx);
        if (idx_lock == tasklet_id) {
            uint32_t tid = 0;
            while (tid < NR_TASKLETS) {
                while (!mutex_trylock(lock_wait_mtx)) {
                    ;
                }
                if (wr_lock_wait[tid] == 1) {
                    tid++;
                }
                mutex_unlock(lock_wait_mtx);
            }
#ifdef KERNEL_ASSERT
            assert(exp == index_exponent);
            if (index_exponent2 != 0) {
                assert(index_exponent == (index_exponent2 + 1));
            }
#endif
#ifndef LAZY_REHASHING
            Bucket* ptr = buckets;
            buckets = buckets2;
            buckets2 = ptr;
            index_exponent2 = index_exponent;
            index_exponent = exp + 1;
#endif
            expand_index_done = 0;
            mutex_lock(index_lock_mtx);
            index_lock = NR_TASKLETS;
            mutex_unlock(index_lock_mtx);
            expand_index(exp, true, expid, tasklet_id);
        }
        else {
            while (idx_lock != NR_TASKLETS) {
                while (!mutex_trylock(index_lock_mtx)) {
                    ;
                }
                idx_lock = index_lock;
                mutex_unlock(index_lock_mtx);
            }
            mutex_lock(lock_wait_mtx);
            wr_lock_wait[tasklet_id] = 0;
            mutex_unlock(lock_wait_mtx);
            expand_index(exp, false, expid, tasklet_id);
        }
        mutex_lock(lock_wait_mtx);
        wr_lock_wait[tasklet_id] = 0;
        mutex_unlock(lock_wait_mtx);
#ifdef KERNEL_ASSERT
        mutex_lock(index_lock_mtx);
        idx_lock = index_lock;
        mutex_unlock(index_lock_mtx);
        assert(idx_lock == NR_TASKLETS);
#endif
        goto GET_LOCKS;
    }

    uint32_t mask = (1u << exp) - 1;
    uint32_t bid1 = bucket_id1(hash_val, mask);
    uint32_t bid2 = bucket_id2(bid1, fgprint, mask);
    uint32_t lpos1 = lock_pos(bid1);
    uint32_t lpos2 = lock_pos(bid2);
    swap_pos(&lpos1, &lpos2);

    acquire_bucket_lock(lpos1);
    mutex_lock(index_exponent_mtx);
    uint32_t curr_exp = index_exponent;
    mutex_unlock(index_exponent_mtx);
    if (curr_exp != exp) {
        release_bucket_lock(lpos1);
        goto GET_LOCKS;
    }
    if (lpos1 != lpos2) {
        acquire_bucket_lock(lpos2);
    }
#ifdef LAZY_REHASHING
    rehash_lock_buckets(lpos1, tasklet_id);
    rehash_lock_buckets(lpos2, tasklet_id);
#endif

    find_bucket_slot(key, bid1, bid2, exp, &buc_slot, tasklet_id);

    if (buc_slot.ret == slot_found) {
        insert_bucket_entry(key, val, fgprint, &buc_slot, tasklet_id);
        release_bucket_lock(lpos1);
        if (lpos1 != lpos2) {
            release_bucket_lock(lpos2);
        }
        return key_inserted;
    }
    else if (buc_slot.ret == duplicate_key) {
        release_bucket_lock(lpos1);
        if (lpos1 != lpos2) {
            release_bucket_lock(lpos2);
        }
        return duplicate_key;
    }
    else if (buc_slot.ret == index_full) {
        /* find_bucket_slot() released the locks
           for expand_conflict and index_full */
        mutex_lock(index_lock_mtx);
        if (index_lock == NR_TASKLETS) {
            index_lock = tasklet_id;
        }
        mutex_unlock(index_lock_mtx);
        goto GET_LOCKS;
    }
    else { /* buc_slot.ret == expand_conflict */
        goto GET_LOCKS;
    }
}

#else /* MULTITASKLET_EXPAND */

OpRet insert(PIMKey_t key, PIMKey_t val, uint32_t tasklet_id) {
    BucketSlot buc_slot;
    uint64_t hash_val = murmur64(key);
    uint8_t fgprint = fingerprint(hash_val);

GET_LOCKS:
    ;
    mutex_lock(index_exponent_mtx);
    uint32_t exp = index_exponent;
    mutex_unlock(index_exponent_mtx);

    mutex_lock(index_lock_mtx);
    uint32_t idx_lock = index_lock;
    mutex_unlock(index_lock_mtx);
    if (idx_lock != NR_TASKLETS) {
        mutex_lock(lock_wait_mtx);
        wr_lock_wait[tasklet_id] = 1;
        mutex_unlock(lock_wait_mtx);
        if (idx_lock == tasklet_id) {
            uint32_t tid = 0;
            while (tid < NR_TASKLETS) {
                while (!mutex_trylock(lock_wait_mtx)) {
                    ;
                }
                if (wr_lock_wait[tid] == 1) {
                    tid++;
                }
                mutex_unlock(lock_wait_mtx);
            }
            expand_index(exp, tasklet_id);
            mutex_lock(index_lock_mtx);
            index_lock = NR_TASKLETS;
            mutex_unlock(index_lock_mtx);         
        }
        else {
            while (idx_lock != NR_TASKLETS) {
                while (!mutex_trylock(index_lock_mtx)) {
                    ;
                }
                idx_lock = index_lock;
                mutex_unlock(index_lock_mtx);
            }
        }
        mutex_lock(lock_wait_mtx);
        wr_lock_wait[tasklet_id] = 0;
        mutex_unlock(lock_wait_mtx);
#ifdef KERNEL_ASSERT
        mutex_lock(index_lock_mtx);
        idx_lock = index_lock;
        mutex_unlock(index_lock_mtx);
        assert(idx_lock == NR_TASKLETS);
#endif
        goto GET_LOCKS;
    }

    uint32_t mask = (1u << exp) - 1;
    uint32_t bid1 = bucket_id1(hash_val, mask);
    uint32_t bid2 = bucket_id2(bid1, fgprint, mask);
    uint32_t lpos1 = lock_pos(bid1);
    uint32_t lpos2 = lock_pos(bid2);
    swap_pos(&lpos1, &lpos2);

    acquire_bucket_lock(lpos1);
    mutex_lock(index_exponent_mtx);
    uint32_t curr_exp = index_exponent;
    mutex_unlock(index_exponent_mtx);
    if (curr_exp != exp) {
        release_bucket_lock(lpos1);
        goto GET_LOCKS;
    }
    if (lpos1 != lpos2) {
        acquire_bucket_lock(lpos2);
    }
#ifdef LAZY_REHASHING
    rehash_lock_buckets(lpos1, tasklet_id);
    rehash_lock_buckets(lpos2, tasklet_id);
#endif

    find_bucket_slot(key, bid1, bid2, exp, &buc_slot, tasklet_id);

    if (buc_slot.ret == slot_found) {
        insert_bucket_entry(key, val, fgprint, &buc_slot, tasklet_id);
        release_bucket_lock(lpos1);
        if (lpos1 != lpos2) {
            release_bucket_lock(lpos2);
        }
        return key_inserted;
    }
    else if (buc_slot.ret == duplicate_key) {
        release_bucket_lock(lpos1);
        if (lpos1 != lpos2) {
            release_bucket_lock(lpos2);
        }
        return duplicate_key;
    }
    else if (buc_slot.ret == index_full) {
        /* find_bucket_slot() released the locks
           for expand_conflict and index_full */
        mutex_lock(index_lock_mtx);
        if (index_lock == NR_TASKLETS) {
            index_lock = tasklet_id;
        }
        mutex_unlock(index_lock_mtx);
        goto GET_LOCKS;
    }
    else { /* buc_slot.ret == expand_conflict */
        goto GET_LOCKS;
    }
}
#endif /* MULTITASKLET_EXPAND */

PIMKey_t search(PIMKey_t key, uint32_t tasklet_id) {
    uint64_t hash_val = murmur64(key);
    uint8_t fgprint = fingerprint(hash_val);

GET_LOCKS:
    ;
    mutex_lock(index_exponent_mtx);
    uint32_t exp = index_exponent;
    mutex_unlock(index_exponent_mtx);

    uint32_t mask = (1u << exp) - 1;
    uint32_t bid1 = bucket_id1(hash_val, mask);
    uint32_t bid2 = bucket_id2(bid1, fgprint, mask);
    uint32_t lpos1 = lock_pos(bid1);
    uint32_t lpos2 = lock_pos(bid2);
    swap_pos(&lpos1, &lpos2);

    acquire_bucket_lock(lpos1);
    mutex_lock(index_exponent_mtx);
    uint32_t curr_exp = index_exponent;
    mutex_unlock(index_exponent_mtx);
    if (curr_exp != exp) {
        release_bucket_lock(lpos1);
        goto GET_LOCKS;
    }
    if (lpos1 != lpos2) {
        acquire_bucket_lock(lpos2);
    }
#ifdef LAZY_REHASHING
    rehash_lock_buckets(lpos1, tasklet_id);
    rehash_lock_buckets(lpos2, tasklet_id);
#endif

    PIMValue_t val = find_key(key, bid1, bid2, tasklet_id);
    release_bucket_lock(lpos1);
    if (lpos1 != lpos2) {
        release_bucket_lock(lpos2);
    }
    return val;
}

int initialization_kernel() {
    uint32_t tasklet_id = me();
    initialize_index(tasklet_id);
    return 0;
}

int insert_kernel() {

    uint32_t tasklet_id = me();

    uint32_t inserted_keys = 0;
    PIMKey_t* wram_keys = (PIMKey_t*) wr_buffer3[tasklet_id];
    PIMKey_t* mram_keys = (PIMKey_t*) (mram_heap + dpu_args.keys_offs);

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
            if (ret != key_inserted) {
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

    *((uint32_t*) wr_buffer2[tasklet_id]) = inserted_keys;
    mutex_lock(lock_wait_mtx);
    wr_lock_wait[tasklet_id] = 1;
    mutex_unlock(lock_wait_mtx);
    barrier_wait(&index_barrier);
    if (tasklet_id == 0) {
        uint32_t num_keys = 0;
        for (uint32_t i = 0; i < NR_TASKLETS; i++) {
            num_keys += *((uint32_t*) wr_buffer2[i]);
        }
        if (num_keys == dpu_args.num_keys) {
            dpu_args.kret = exec_success;
        }
        else {
            dpu_args.kret = count_mismatch;
        }
    }

    return 0;
}

int search_kernel() {

    uint32_t tasklet_id = me();

    uint32_t found = 0;
    uint32_t not_found = 0;
    PIMKey_t* wram_keys = (PIMKey_t*) wr_buffer3[tasklet_id];
    PIMKey_t* mram_keys = (PIMKey_t*) (mram_heap + dpu_args.keys_offs);

    for (uint32_t i = tasklet_id * NR_KEYS_PER_WRAM_BUFFER; i < dpu_args.num_keys; i += NR_TASKLETS * NR_KEYS_PER_WRAM_BUFFER) {
        mram_read((__mram_ptr const void*) &mram_keys[i], wram_keys, WRAM_BUFFER_SIZE);
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
        mram_write(wram_keys, (__mram_ptr void*) &mram_keys[i], WRAM_BUFFER_SIZE);
    }

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
        if (num_not_found == 0) {
            dpu_args.kret = exec_success;
        }
        else {
            dpu_args.kret = count_mismatch;
        }
    }

    return 0;
}

int mapping_kernel() {

    uint32_t tasklet_id = me();

    if (tasklet_id == 0) {
        prefix = 0;
        wr_hist = (uint32_t*)wr_bucket_locks;
        wr_part_buffer = (PIMKey_t*)((char*)wr_pool + 8 * KiB);
        wr_part_counter = (uint8_t*)((char*)wr_pool + 28 * KiB);
    }
    barrier_wait(&index_barrier);

    /* cache locks in MRAM */
    uint32_t locks_per_buffer = WRAM_BUFFER_SIZE / sizeof(BucketLock); /* TODO: ensure locks fit buffer precisely */
    BucketLock* mr_bucket_locks = (BucketLock*)((char*)mram_heap + dpu_args.keys_offs + 5 * MiB);
    for (uint32_t i = tasklet_id * locks_per_buffer; i < NR_BUCKET_LOCKS; i += NR_TASKLETS * locks_per_buffer) {
        mram_write(&wr_bucket_locks[i], (__mram_ptr void*) &mr_bucket_locks[i], WRAM_BUFFER_SIZE);
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
    PIMKey_t* mr_inkeys = (PIMKey_t*)((char*)mram_heap + dpu_args.keys_offs);
    PIMKey_t* mr_outkeys = (PIMKey_t*)((char*)mram_heap + dpu_args.keys_offs + 6 * MiB);
    uint32_t* mr_hist = (uint32_t*)((char*)mram_heap + dpu_args.keys_offs);

    for (uint32_t i = tasklet_id * keys_per_buffer; i < dpu_args.num_keys; i += NR_TASKLETS * keys_per_buffer) {
        mram_read((__mram_ptr void const*) &mr_inkeys[i], wr_keys, WRAM_BUFFER_SIZE);
        uint32_t num_keys = ((i + keys_per_buffer) < dpu_args.num_keys) ?
                            (keys_per_buffer) :
                            (dpu_args.num_keys - i);
        for (uint32_t j = 0; j < num_keys; j++) {
            PIMKey_t key = wr_keys[j];
            uint32_t part = superfast(key) % NR_DPUS;
#ifdef MUTEX_POOL
            mutex_pool_lock(&bucket_locks_mtx, part); /* borrowing the mutex pool */
            wr_hist[part]++;
            mutex_pool_unlock(&bucket_locks_mtx, part);
#elif defined(VIRTUAL_MUTEX)
            vmutex_lock(&bucket_locks_vmtx, part);
            wr_hist[part]++;
            vmutex_unlock(&bucket_locks_vmtx, part);
#endif
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

    for (uint32_t i = tasklet_id * keys_per_buffer; i < dpu_args.num_keys; i += NR_TASKLETS * keys_per_buffer) {
        mram_read((__mram_ptr void const*) &mr_inkeys[i], wr_keys, WRAM_BUFFER_SIZE);
        uint32_t num_keys = ((i + keys_per_buffer) < dpu_args.num_keys) ?
                            (keys_per_buffer) :
                            (dpu_args.num_keys - i);
        for (uint32_t j = 0; j < num_keys; j++) {
            PIMKey_t key = wr_keys[j];
            uint32_t part = superfast(key) % NR_DPUS;
            uint32_t mr_offs;
#ifdef MUTEX_POOL
            mutex_pool_lock(&bucket_locks_mtx, part);
#elif defined(VIRTUAL_MUTEX)
            vmutex_lock(&bucket_locks_vmtx, part);
#endif
            if (wr_part_counter[part] == 0) {
                wr_part_buffer[2 * part] = key;
                wr_part_counter[part] = 1;
#ifdef MUTEX_POOL
                mutex_pool_unlock(&bucket_locks_mtx, part);
#elif defined(VIRTUAL_MUTEX)
                vmutex_unlock(&bucket_locks_vmtx, part);
#endif
            }
            else {
                wr_part_buffer[2 * part + 1] = key;
                mr_offs = wr_hist[part];
                wr_hist[part] += 2;
                wr_part_counter[part] = 0;
#ifdef MUTEX_POOL
                mutex_pool_unlock(&bucket_locks_mtx, part);
#elif defined(VIRTUAL_MUTEX)
                vmutex_unlock(&bucket_locks_vmtx, part);
#endif
                mram_write(&wr_part_buffer[2 * part], (__mram_ptr void*) &mr_outkeys[mr_offs], 2 * KEY_SIZE);
            }
        }
    }
    barrier_wait(&index_barrier);

    for (uint32_t part = tasklet_id; part < NR_DPUS; part += NR_TASKLETS) {
        if (wr_part_counter[part] == 1) {
            uint32_t mr_offs = wr_hist[part];
            wr_part_buffer[2 * part + 1] = 0; /* padding key */
            mram_write(&wr_part_buffer[2 * part], (__mram_ptr void*) &mr_outkeys[mr_offs], 2 * KEY_SIZE);
            wr_hist[part] += 2;
            wr_part_counter[part] = 0;
        }
    }
    barrier_wait(&index_barrier);

    uint32_t hist_elem_per_buffer = WRAM_BUFFER_SIZE / sizeof(uint32_t);
    for (uint32_t i = tasklet_id * hist_elem_per_buffer; i < NR_DPUS; i += NR_TASKLETS * hist_elem_per_buffer) {
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
    barrier_wait(&index_barrier);

    /* retrieve cached locks back to WRAM */
    for (uint32_t i = tasklet_id; i < NR_BUCKET_LOCKS; i += NR_TASKLETS) {
        mram_read((__mram_ptr const void*) &mr_bucket_locks[i], &wr_bucket_locks[i], WRAM_BUFFER_SIZE);
    }

    return 0;
}

int kvmapping_kernel() {

    uint32_t tasklet_id = me();

    if (tasklet_id == 0) {
        prefix = 0;
        wr_hist = (PIMKey_t*)((char*)wr_pool + 8 * KiB);
    }
    barrier_wait(&index_barrier);

    for (uint32_t i = tasklet_id; i < NR_DPUS; i += NR_TASKLETS) {
        wr_hist[i] = 0;
    }
    barrier_wait(&index_barrier);

    uint32_t keys_per_buffer = WRAM_BUFFER_SIZE / sizeof(PIMKey_t);
    PIMKey_t* wr_keys = (PIMKey_t*) wr_buffer[tasklet_id];
    PIMKey_t* mr_inkeys = (PIMKey_t*)((char*)mram_heap + dpu_args.keys_offs);
    PIMKey_t* mr_outkeys = (PIMKey_t*)((char*)mram_heap + dpu_args.keys_offs + 6 * MiB);
    uint32_t* mr_hist = (uint32_t*)((char*)mram_heap + dpu_args.keys_offs);

    for (uint32_t i = tasklet_id * keys_per_buffer; i < (2 * dpu_args.num_keys); i += NR_TASKLETS * keys_per_buffer) {
        mram_read((__mram_ptr void const*) &mr_inkeys[i], wr_keys, WRAM_BUFFER_SIZE);
        uint32_t num_keys = ((i + keys_per_buffer) < (2 * dpu_args.num_keys)) ?
                            (keys_per_buffer) :
                            ((2 * dpu_args.num_keys) - i);
        for (uint32_t j = 0; j < num_keys; j += 2) {
            PIMKey_t key = wr_keys[j];
            uint32_t part = superfast(key) % NR_DPUS;
#ifdef MUTEX_POOL
            mutex_pool_lock(&bucket_locks_mtx, part); /* borrowing the mutex pool */
            wr_hist[part]++;
            mutex_pool_unlock(&bucket_locks_mtx, part);
#elif defined(VIRTUAL_MUTEX)
            vmutex_lock(&bucket_locks_vmtx, part);
            wr_hist[part]++;
            vmutex_unlock(&bucket_locks_vmtx, part);
#endif
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

    for (uint32_t i = tasklet_id * keys_per_buffer; i < (2 * dpu_args.num_keys); i += NR_TASKLETS * keys_per_buffer) {
        mram_read((__mram_ptr void const*) &mr_inkeys[i], wr_keys, WRAM_BUFFER_SIZE);
        uint32_t num_keys = ((i + keys_per_buffer) < (2 * dpu_args.num_keys)) ?
                            (keys_per_buffer) :
                            ((2* dpu_args.num_keys) - i);
        for (uint32_t j = 0; j < num_keys; j += 2) {
            PIMKey_t key = wr_keys[j];
            uint32_t part = superfast(key) % NR_DPUS;
#ifdef MUTEX_POOL
            mutex_pool_lock(&bucket_locks_mtx, part);
            uint32_t mr_offs = wr_hist[part];
            wr_hist[part]++;
            mutex_pool_unlock(&bucket_locks_mtx, part);
#elif defined(VIRTUAL_MUTEX)
            vmutex_lock(&bucket_locks_vmtx, part);
            uint32_t mr_offs = wr_hist[part];
            wr_hist[part]++;
            vmutex_unlock(&bucket_locks_vmtx, part);
#endif
            mram_write(&wr_keys[j], (__mram_ptr void*) &mr_outkeys[2 * mr_offs], 2 * KEY_SIZE);
        }
    }
    barrier_wait(&index_barrier);

    uint32_t hist_elem_per_buffer = WRAM_BUFFER_SIZE / sizeof(uint32_t);
    for (uint32_t i = tasklet_id * hist_elem_per_buffer; i < NR_DPUS; i += NR_TASKLETS * hist_elem_per_buffer) {
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
        uint32_t count = 0;
        uint32_t num_keys = 0;
        uint32_t num_buckets = (1u << index_exponent);
        for (uint32_t i = 0; i < num_bucket_locks; i++) {
            count = wr_bucket_locks[i].key_count;
            num_keys += count;
        }
#ifdef KERNEL_ASSERT
        assert(num_keys >= 0);
#endif
        dpu_args.num_buckets = num_buckets;
        dpu_args.num_keys = num_keys;
    }

    return 0;
}

int (*kernels[NR_KERNELS])() = {initialization_kernel, insert_kernel,
                                search_kernel, mapping_kernel,
                                kvmapping_kernel, mem_utilization_kernel};

int main() {
    return kernels[dpu_args.kernel]();
}
