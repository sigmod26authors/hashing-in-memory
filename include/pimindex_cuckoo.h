#ifndef pimindex_cuckoo_h_
#define pimindex_cuckoo_h_

#define KiB (1 << 10)
#define MiB (KiB << 10)

#define DIVCEIL(n, d) (((n) - 1) / (d) + 1)

#define NR_DPUS 64
#define NR_TASKLETS 16
#define NR_OPERATIONS 1 * 1000000
#define NR_LOADS 0 * 1000000

#define NR_KERNELS 6
#define MAX_INDEX_SPACE (24 * MiB)
#define WRAM_BUFFER_SIZE 512

#define ENTRIES_PER_BUCKET_PIM 4
#define BUCKET_ENTRY_SIZE sizeof(BucketEntry)
#define BUCKET_SIZE sizeof(Bucket)
#define MAX_NUM_BUCKETS (MAX_INDEX_SPACE / BUCKET_SIZE)
#define MAX_SIZE_BUCKET_LOCKS (12 * KiB)
#define MAX_NUM_BUCKET_LOCKS (MAX_SIZE_BUCKET_LOCKS / sizeof(BucketLock))
#define NR_BUCKET_LOCKS 2048
#define BUCKET_HEADER_SKIP 2

#define MAX_HOPS 2
#define MAX_CUCKOO_COUNT 10
#define MIN_LOAD_FACTOR (0.05)
#define INIT_INDEX_EXPONENT 16
#define MAX_INDEX_EXPONENT 19

#define CHUNK_SIZE 1024
#define MAX_NUM_CHUNKS ((48 * MiB) / CHUNK_SIZE)

#define DPU_PROFILE "sgXferEnable=true"
#define INDEX_INSERT_BIN "./../dpu_bin/dpu_pimindex"
// #define CSV_FILE "./../res/res.csv"
// #define DATA_FILE "./../data/data1M"

typedef uint32_t PIMKey_t;
typedef uint32_t PIMValue_t;
typedef struct pimindex_dpu_args_t pimindex_dpu_args_t;
typedef struct data_xfer_kernel_args_t data_xfer_kernel_args_t;
typedef struct sg_partition_xfer_args_t sg_partition_xfer_args_t;
typedef struct BucketHeader BucketHeader;
typedef struct BucketEntry BucketEntry;
typedef struct BucketLock BucketLock;
typedef struct BucketSlot BucketSlot;
typedef struct CuckooEntry CuckooEntry;
typedef struct CuckooSlot CuckooSlot;
typedef struct CuckooSlotQueue CuckooSlotQueue;

#define DEFAULT_VALUE (PIMValue_t)1
#define KEY_SIZE sizeof(PIMKey_t)
#define NR_KEYS_PER_WRAM_BUFFER (WRAM_BUFFER_SIZE / KEY_SIZE)

#define RESULTS_CHECK
#define MEM_UTILIZATION
#define KERNEL_ASSERT
#define MUTEX_POOL
// #define VIRTUAL_MUTEX
#define LAZY_REHASHING
#define MULTITASKLET_EXPAND

typedef enum KernelRet {
    exec_success,
    not_unique,
    insert_failure,
    count_mismatch,
} KernelRet;

struct pimindex_dpu_args_t {
    uint32_t index_offs;
    union {
        uint32_t kernel;
        KernelRet kret;
    };
    union {
        uint32_t num_buckets;
        uint32_t keys_offs;
    };
    union {
        uint32_t num_keys;
        uint32_t res_compl;
    };
};

struct data_xfer_kernel_args_t {
    uint32_t output_offset;
    uint32_t num_elems;
};

struct sg_partition_xfer_args_t {
    uint32_t* num_partitions;
    uint32_t** partition_sizes;
    PIMKey_t*** partition_ptrs;
};

struct BucketHeader {
    uint8_t bitmap[8];
    uint8_t fingerprints[8];
};

struct BucketEntry {
    uint32_t key;
    uint32_t val;
};

typedef union Bucket {
    BucketEntry entries[BUCKET_HEADER_SKIP + ENTRIES_PER_BUCKET_PIM];
    BucketHeader header;
} Bucket;

struct BucketLock {
    uint32_t locked : 1;
    uint32_t rehashed : 1;
    uint32_t key_count : 30;
};

typedef enum OpRet {
    slot_found,
    key_inserted,
    key_found,
    duplicate_key,
    index_full,
    expand_conflict,
} OpRet;

struct BucketSlot {
    uint32_t bid;
    uint32_t slot;
    OpRet ret;
};

struct CuckooEntry {
    uint32_t bid;
    uint32_t slot;
    uint64_t hash;
    uint8_t fgprint;
};

struct CuckooSlot {
    uint32_t bid;
    uint8_t pathcode;
    int8_t hops;
};

struct CuckooSlotQueue {
    CuckooSlot slots[MAX_CUCKOO_COUNT];
    uint32_t first;
    uint32_t last;
};


#endif
