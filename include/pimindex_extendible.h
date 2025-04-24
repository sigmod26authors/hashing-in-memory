#ifndef pimindex_extendible_h_
#define pimindex_extendible_h_

#define RESULTS_CHECK
#define MEM_UTILIZATION
#define BUCKET128

#define KiB (1 << 10)
#define MiB (KiB << 10)

#define DIVCEIL(n, d) (((n) - 1) / (d) + 1)

#define NR_DPUS 64
#define NR_TASKLETS 16
#define NR_OPERATIONS 1 * 1000000
#define NR_LOADS 0 * 1000000

#define NR_KERNELS 7
#define MAX_MRAM_SIZE (48 * MiB)
#define WRAM_BUFFER_SIZE 512

#ifdef BUCKET64
#define BUCKET_HEADER_SKIP 1
#define ENTRIES_PER_BUCKET_PIM 7
#define BUCKETS_PER_CHUNK 2048
#elif defined(BUCKET128)
#define BUCKET_HEADER_SKIP 2
#define ENTRIES_PER_BUCKET_PIM 14
#define BUCKETS_PER_CHUNK 1024
#elif defined(BUCKET256)
#define BUCKET_HEADER_SKIP 4
#define ENTRIES_PER_BUCKET_PIM 28
#define BUCKETS_PER_CHUNK 512
#elif defined(BUCKET512)
#define BUCKET_HEADER_SKIP 8
#define ENTRIES_PER_BUCKET_PIM 56
#define BUCKETS_PER_CHUNK 256
#endif
#define BUCKET_ENTRY_SIZE sizeof(BucketEntry)
#define BUCKET_SIZE sizeof(Bucket)

#define CHUNK_SIZE (BUCKETS_PER_CHUNK * BUCKET_SIZE)
#define MAX_NUM_CHUNKS (MAX_MRAM_SIZE / CHUNK_SIZE)
#define BITMAP_LEN DIVCEIL(MAX_NUM_CHUNKS, 32)
#define INIT_GLOBAL_DEPTH 3

#define BUCKET_LOCKS_PER_CHUNK (1u << 3)
#define NR_BUCKET_LOCKS (BUCKET_LOCKS_PER_CHUNK * MAX_NUM_CHUNKS)
#define LOCKS_PER_CHUNK ((1u << 3) + 1)
#define LOCK_TABLE_LEN (MAX_NUM_CHUNKS * LOCKS_PER_CHUNK)

#define DPU_PROFILE "sgXferEnable=true"
#define INDEX_INSERT_BIN "./../dpu_bin/dpu_pimindex"
// #define CSV_FILE "./../res/res.csv"
// #define DATA_FILE "./../data/data1M"

typedef uint32_t PIMKey_t;
typedef uint32_t PIMValue_t;
typedef struct pimindex_dpu_args_t pimindex_dpu_args_t;
typedef struct data_xfer_kernel_args_t data_xfer_kernel_args_t;
typedef struct BucketHeader BucketHeader;
typedef struct BucketEntry BucketEntry;
typedef struct ChunkHeader ChunkHeader;
typedef struct IndexFile IndexFile;
typedef struct CBLock CBLock;

#define DEFAULT_VALUE (PIMValue_t)1
#define KEY_SIZE sizeof(PIMKey_t)
#define NR_KEYS_PER_WRAM_BUFFER (WRAM_BUFFER_SIZE / KEY_SIZE)

typedef enum KernelRet {
    exec_success,
    not_unique,
    insert_failure,
    count_mismatch,
} KernelRet;

struct pimindex_dpu_args_t {
    uint32_t index_offs;
    union {
        uint32_t num_chunks;
        uint32_t kernel;
    };
    union {
        uint32_t num_buckets;
        uint32_t keys_offs;
    };
    union {
        uint32_t num_keys;
        KernelRet kret;
    };
};

struct data_xfer_kernel_args_t {
    uint32_t output_offset;
    uint32_t num_elems;
};

struct BucketHeader {
#ifdef BUCKET64
    uint8_t bitmap; /* [7:7] reserved, [6:0] bitmap */
#elif defined(BUCKET128)
    uint16_t bitmap; /* [15:14] reserved, [13:0] bitmap */
#elif defined(BUCKET256)
    uint32_t bitmap; /* [31:28] reserved, [27:0] bitmap */
#elif defined(BUCKET512)
    uint32_t bitmap[2]; /* [63:56] reserved, [55:0] bitmap */
#endif
    uint8_t fingerprints[ENTRIES_PER_BUCKET_PIM];
};

struct BucketEntry {
    uint32_t key;
    uint32_t val;
};

typedef union Bucket {
    BucketEntry entries[BUCKET_HEADER_SKIP +
                        ENTRIES_PER_BUCKET_PIM];
    BucketHeader header;
} Bucket;

struct ChunkHeader {
    uint8_t local_depth;
    uint8_t hash_scheme;
    uint16_t reserve;
    uint32_t cid;
};

struct IndexFile {
    ChunkHeader chunk_headers[0];
};

struct CBLock {
    uint32_t tasklet_id : 8;
    uint32_t ref_count : 8;
    uint32_t version : 16;
};

typedef enum OpRet {
    entry_inserted,
    update_conflict,
    duplicate_key,
    bucket_full,
} OpRet;

struct sg_partition_xfer_args_t {
    uint32_t* num_partitions;
    uint32_t** partition_sizes;
    PIMKey_t*** partition_ptrs;
};


#endif
