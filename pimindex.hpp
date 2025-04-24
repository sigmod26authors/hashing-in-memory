#ifndef pimindex_hpp_
#define pimindex_hpp_

#include <iostream>
#include <random>
#include <map>
#include <fstream>
#include <cassert>
#include <chrono>
#include <cstring>
#include <unistd.h>

#include <dpu>
#include <dpu_log.h>

#include "pimindex.h"


/* scatter-gather transfer */
bool sg_partition_func(struct sg_block_info* cpu_buffer, uint32_t dpuid, uint32_t part, void* params);
void* multidim_malloc(uint32_t rows, std::vector<std::vector<PIMKey_t*>*>& ptrs);

/* data generation */
void randomize_keys(PIMKey_t* workload, uint32_t len);
void* generate_workload(uint32_t num_keys);
void generate_keys(void *workload, uint32_t num_keys);

/* utilities */
uint32_t get_type_size(uint32_t type);

/* MRAM management */
struct mram_heap_obj_t {
    mram_heap_obj_t();

    uint32_t offset;
    uint32_t max_block_size;
    std::vector<uint32_t> block_sizes;
    uint32_t elem_type;
    uint32_t elem_size;
    std::string name;
};

struct mram_heap_t {
    std::vector<mram_heap_obj_t*> objs;
};

struct mram_mem_mgr_t {
    mram_mem_mgr_t();
    ~mram_mem_mgr_t();

    mram_heap_obj_t* alloc_block(const std::string &name,
        std::vector<uint32_t>& block_sizes, uint32_t max_block_size,
            uint32_t align_size, uint32_t type);
    uint32_t get_slot(mram_heap_obj_t* obj, uint32_t max_block_size,
                                                    uint32_t align_size);
    mram_heap_obj_t* get_block(std::string &name);
    void free_block(std::string &name);
    void print_mram_info();

    mram_heap_t* heap_;
    std::vector<uint32_t> offsets_;
    std::vector<std::pair<bool, mram_heap_obj_t*>> slots_;
};

/* timer */
using time_val = std::chrono::_V2::steady_clock::time_point;

struct event {
    std::vector<time_val> starts;
    std::vector<time_val> stops;
    std::string name;
};

struct timer {
    void start(const std::string &name);
    void stop();
    void print();
    void print_to_csv(const std::string &f,
            const std::string &mark, bool append = true);

private:
    std::map<std::string, event> events;
    std::vector<std::string> events_seq;
    bool timing = false;
    std::string cur_event;
};

/* PIM executions */
void transfer_data();
void run_cooperative(void* workload, uint32_t load_keys, uint32_t insert_keys);
void run_three_level(void* workload, uint32_t load_keys, uint32_t insert_keys);
void run_pimindex_cuckoo(void* workload, uint32_t load_keys, uint32_t insert_keys);
void run_pimindex_cuckoo_batch(void* workload, uint32_t load_keys, uint32_t insert_keys);
void run_pimindex_extendible(void* workload, uint32_t load_keys, uint32_t insert_keys);
void run_pimindex_extendible_batch(void* workload, uint32_t load_keys, uint32_t insert_keys);
void run_pimindex_direct_mapping(void* workload, uint32_t load_keys, uint32_t insert_keys);

/* print */
#define ANSI_RED        "\033[31m"
#define ANSI_GREEN      "\033[32m"
#define ANSI_MAGENTA    "\033[35m"
#define ANSI_RESET      "\033[0m"

#ifdef VERBOSE
#define PRINT_MSG(fmt, ...)         printf(fmt "\n", ##__VA_ARGS__)
#define PRINT_TOP_RULE              printf("%s===============%s\n", ANSI_GREEN, ANSI_RESET);
#define PRINT_INFO(fmt, ...)        fprintf(stdout, "%sINFO:       %s" fmt "\n", ANSI_GREEN, ANSI_RESET, ##__VA_ARGS__)
#define PRINT_WARNING(fmt, ...)     fprintf(stderr, "%sWARNING:    %s" fmt "\n", ANSI_MAGENTA, ANSI_RESET, ##__VA_ARGS__)
#define PRINT_ERROR(fmt, ...)       fprintf(stderr, "%sERROR:      %s" fmt "\n", ANSI_RED, ANSI_RESET, ##__VA_ARGS__)
#else
#define PRINT_MSG(fmt, ...)
#define PRINT_TOP_RULE
#define PRINT_INFO(fmt, ...)        printf(fmt "\n", ##__VA_ARGS__)
#define PRINT_WARNING(fmt, ...)     printf(fmt "\n", ##__VA_ARGS__)
#define PRINT_ERROR(fmt, ...)       printf(fmt "\n", ##__VA_ARGS__)
#endif

#endif
