#ifndef three_level_h_
#define three_level_h_

#define RESULTS_CHECK
#define MEM_UTILIZATION
// #define PINNED_CHUNK_HEADERS
#define KERNEL_PROFILING
#define THREE_LEVEL_PARTITIONING
#define HASH32

#define KiB (1 << 10)
#define MiB (KiB << 10)

#define DIVCEIL(n, d) (((n) - 1) / (d) + 1)

#define NR_DPUS 64
#define NR_TASKLETS 16
#define NR_OPERATIONS 1 * 1000000
#define NR_LOADS 0 * 1000000

#define NR_KERNELS 7
#define MAX_MRAM_SIZE (48 * MiB)
#define MAX_MRAM_SIZE_PER_TASKLET (MAX_MRAM_SIZE / NR_TASKLETS)
#define MRAM_INDEX_OFFSET 0
#define MRAM_INPUT_OFFSET MAX_MRAM_SIZE
#define MRAM_OUTPUT_OFFSET (MAX_MRAM_SIZE + (6 * MiB))
#define MRAM_LOG_OFFSET (MAX_MRAM_SIZE + (13 * MiB))
#define WRAM_BUFFER_SIZE 512

#define BUCKET_HEADER_SKIP 2
#define ENTRIES_PER_BUCKET_PIM 14
#define BUCKET_ENTRY_SIZE sizeof(BucketEntry)
#define BUCKET_SIZE sizeof(Bucket)
#define KEY_SIZE sizeof(PIMKey_t)

#define BUCKETS_PER_CHUNK 1024
#define NR_KEYS_PER_WRAM_BUFFER (WRAM_BUFFER_SIZE / KEY_SIZE)
#define CHUNK_SIZE (BUCKETS_PER_CHUNK * BUCKET_SIZE)
#define MAX_NUM_CHUNKS (MAX_MRAM_SIZE / CHUNK_SIZE)
#define BITMAP_LEN DIVCEIL(MAX_NUM_CHUNKS, 32)
#define MAX_NUM_CHUNKS_PER_TASKLET (MAX_MRAM_SIZE_PER_TASKLET / CHUNK_SIZE)
#define BITMAP_LEN_PER_TASKLET DIVCEIL(MAX_NUM_CHUNKS_PER_TASKLET, 32)
#define INIT_GLOBAL_DEPTH 3
#define DEFAULT_VALUE (PIMValue_t)1

#define BUCKET_LOCKS_PER_CHUNK (1u << 3)
#define NR_BUCKET_LOCKS (BUCKET_LOCKS_PER_CHUNK * MAX_NUM_CHUNKS)
#define LOCKS_PER_CHUNK ((1u << 3) + 1)
#define LOCK_TABLE_LEN (MAX_NUM_CHUNKS * LOCKS_PER_CHUNK)

#define DPU_PROFILE "sgXferEnable=true"
#define PIMINDEX_BIN1 "./../dpu_bin/dpu_pimindex"
#define PIMINDEX_BIN2 "./../dpu_bin/dpu_pimindex2"
// #define CSV_FILE "./../res/res.csv"
// #define DATA_FILE "./data/eval/data1M"

typedef uint32_t PIMKey_t;
typedef uint32_t PIMValue_t;
#ifdef HASH32
typedef uint32_t HashValue_t;
#else
typedef uint64_t HashValue_t;
#endif
typedef struct pimindex_dpu_args_t pimindex_dpu_args_t;
typedef struct data_xfer_kernel_args_t data_xfer_kernel_args_t;
typedef struct BucketHeader BucketHeader;
typedef struct BucketEntry BucketEntry;
typedef struct ChunkHeader ChunkHeader;
typedef struct IndexFile IndexFile;
typedef struct CBLock CBLock;


typedef enum KernelRet {
    exec_success,
    not_unique,
    insert_failure,
    count_mismatch,
} KernelRet;

struct pimindex_dpu_args_t {
    uint32_t log_level;
    union {
        uint32_t num_chunks;
        uint32_t kernel;
    };
    union {
        uint32_t num_buckets;
        uint32_t padding;
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

typedef enum elem_type_t {
    UINT32,
    UINT64,
    BUCKET
} elem_type_t;

struct BucketHeader {
    uint16_t bitmap; /* [15:14] reserved, [13:0] bitmap */
    uint8_t fingerprints[ENTRIES_PER_BUCKET_PIM];
};

struct BucketEntry {
    uint32_t key;
    uint32_t val;
};

typedef union Bucket {
    BucketEntry entries[2 + ENTRIES_PER_BUCKET_PIM];
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
