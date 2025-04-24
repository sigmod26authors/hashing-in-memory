#include <thread>
#include <mutex>

#include "pimindex.hpp"
#include "hash.h"


std::mutex dpu_buffer_mutexes[NR_DPUS];

struct cpu_thread_arg_t {
    uint32_t beg;
    uint32_t end;
    bool insert;
    PIMKey_t* workload;
    std::vector<std::vector<PIMKey_t>*>* buffers;
};

void copy_keys(cpu_thread_arg_t* params) {
    uint32_t beg = params->beg;
    uint32_t end = params->end;
    bool ins = params->insert;
    PIMKey_t* workload = params->workload;
    std::vector<std::vector<PIMKey_t>*>* buffers = params->buffers;
    for (uint32_t i = beg; i < end; i++) {
        PIMKey_t key = workload[i];
        // uint32_t id = ((NR_DPUS - 1) & superfast(key));
        uint32_t id = superfast(key) % NR_DPUS;
        std::lock_guard<std::mutex> guard(dpu_buffer_mutexes[id]);
        (*buffers)[id]->push_back(key);
        if (ins) {
            (*buffers)[id]->push_back(DEFAULT_VALUE);
        }
    }
}

void run_pimindex_direct_mapping(void* workload, uint32_t load_keys, uint32_t insert_keys) {

    try {

        /* allocate DPUs and load binary */
        uint32_t num_dpus, dpuid;
        struct dpu_set_t dpu_set, dpu;

        DPU_ASSERT(dpu_alloc(NR_DPUS, NULL, &dpu_set));
        DPU_ASSERT(dpu_get_nr_dpus(dpu_set, &num_dpus));
        assert(num_dpus == NR_DPUS);
        DPU_ASSERT(dpu_load(dpu_set, INDEX_INSERT_BIN, NULL));

        PRINT_TOP_RULE
        PRINT_INFO("PIM INDEX: Direct Mapping");
        PRINT_TOP_RULE
        PRINT_MSG("DPUs: %u", NR_DPUS);
        PRINT_MSG("Tasklets: %u", NR_TASKLETS);
        PRINT_MSG("Index entry size: %lu", BUCKET_ENTRY_SIZE);
        PRINT_MSG("Key size: %lu", KEY_SIZE);
        PRINT_MSG("Bucket size: %lu", BUCKET_SIZE);
        PRINT_MSG("Buckets per chunk: %u", BUCKETS_PER_CHUNK);
        PRINT_MSG("Chunk size: %.*f KiB", 2, (CHUNK_SIZE/(float)KiB));
        PRINT_MSG("Max. chunks: %lu", MAX_NUM_CHUNKS);
        PRINT_MSG("Chunks bitmap length: %lu", BITMAP_LEN);
        PRINT_MSG("Chunks bitmap size: %lu B", BITMAP_LEN * sizeof(uint32_t));
        /*PRINT_MSG("Lock table length: %lu", LOCK_TABLE_LEN);
        PRINT_MSG("Lock table size: %.*f KiB", 2,
            ((LOCK_TABLE_LEN * sizeof(CBLock))/(float)KiB));*/
        PRINT_MSG("Chunk header lock size: %.*f KiB", 2,
            ((512 * sizeof(uint8_t))/(float)KiB));
        PRINT_MSG("Chunk lock size: %.*f KiB", 2,
            ((MAX_NUM_CHUNKS * sizeof(CBLock))/(float)KiB));
        PRINT_MSG("Bucket locks per chunk: %u", BUCKET_LOCKS_PER_CHUNK);
        PRINT_MSG("Bucket locks: %lu", NR_BUCKET_LOCKS);
        PRINT_MSG("Bucket lock size: %.*f KiB", 2,
            ((NR_BUCKET_LOCKS * sizeof(CBLock))/(float)KiB));

        timer t;

        bool validation;
        PIMKey_t *insert_workload;
        PIMKey_t *search_workload;

        mram_mem_mgr_t *mram_mgr = new mram_mem_mgr_t();
        std::vector<pimindex_dpu_args_t> dpu_params(NR_DPUS);
        std::vector<uint32_t> chunk_sizes(NR_DPUS, CHUNK_SIZE);
        std::vector<mram_heap_obj_t*> chunk_blocks(MAX_NUM_CHUNKS);
        /* TODO: struct for instances of the same block type */
        std::vector<std::vector<PIMKey_t>*> dpu_input_buffers(NR_DPUS);
        std::vector<uint32_t>* zeros =
            new std::vector<uint32_t>(((63 * MiB) / sizeof(uint32_t)), 0);

        /* zero out MRAM in all DPUs */
        PRINT_MSG("Zero out MRAM of all DPUs...");
        DPU_ASSERT(dpu_broadcast_to(dpu_set, DPU_MRAM_HEAP_POINTER_NAME,
                                0, zeros->data(), 63 * MiB, DPU_XFER_DEFAULT));

        /* allocate max. hash index chunks in MRAM */
        PRINT_MSG("Allocate max. hash index chunks in MRAM...");
        for (uint32_t i = 0; i < MAX_NUM_CHUNKS; i++) {
            chunk_blocks[i] =
                mram_mgr->alloc_block(("HT Chunk " + std::to_string(i)),
                                        chunk_sizes, CHUNK_SIZE, 8, 2);
        }

        // mram_mgr->print_mram_info();

        /* transfer hash index initialization parameters to DPUs */
        PRINT_MSG("Transfer hash index initialization parameters to DPUs...");
        DPU_FOREACH(dpu_set, dpu, dpuid) {
            dpu_params[dpuid].index_offs = chunk_blocks[0]->offset;
            dpu_params[dpuid].kernel = 0;
            dpu_params[dpuid].keys_offs = 0;
            dpu_params[dpuid].num_keys = 0;
            DPU_ASSERT(dpu_prepare_xfer(dpu, &dpu_params[dpuid]));
        }
        DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU,
            "dpu_args", 0, sizeof(pimindex_dpu_args_t), DPU_XFER_DEFAULT));

        /* launch the index initialization kernel */
        PRINT_MSG("Executing the index initialization kernel on DPUs...");
        t.start("DPU exec (index init.)");
        DPU_ASSERT(dpu_launch(dpu_set, DPU_SYNCHRONOUS));
        t.stop();

#ifdef PRINT_DPU_LOGS
        /* dump DPU logs */
        PRINT_MSG("Dump DPU logs...");
        DPU_FOREACH(dpu_set, dpu, dpuid) {
            DPU_ASSERT(dpu_log_read(dpu, stdout));
        }
#endif

#ifdef LOAD_INDEX
        /* TODO */
        PIMKey_t *load_workload = (PIMKey_t*) workload;
        uint32_t max_load_keys_per_dpu = 0;
        uint32_t min_load_keys_per_dpu = (uint32_t)(-1);
        uint32_t avg_load_keys_per_dpu = DIVCEIL(load_keys, NR_DPUS);
        uint32_t avg_size_load_keys_per_dpu = avg_load_keys_per_dpu * KEY_SIZE;
        uint32_t load_keys_offs = MAX_MRAM_SIZE + (2 * MiB);
        PRINT_TOP_RULE
        PRINT_MSG("Load Keys: %.*f M", 2, load_keys/(float)1000000);
        PRINT_MSG("Load size: %.*f MiB", 2, (load_keys * KEY_SIZE)/(float)MiB);
        PRINT_MSG("Average load keys per DPU: %u", avg_load_keys_per_dpu);
        PRINT_MSG("Average load size per DPU: %.*f MiB", 2, (avg_load_keys_per_dpu * KEY_SIZE)/(float)MiB);

        /* preprocess load keys for each DPU */
        for (uint32_t i = 0; i < NR_DPUS; i++) {
            dpu_input_buffers[i] = new std::vector<PIMKey_t>();
        }

        PRINT_MSG("Preprocess load keys...");
        t.start("CPU exec (preproc. load keys)");
        for (uint32_t i = 0; i < load_keys; i++) {
            PIMKey_t key = load_workload[i];
            uint32_t dpuid = ((NR_DPUS - 1) & key_to_dpu_hash(key));
            dpu_input_buffers[dpuid]->push_back(key);
        }
        t.stop();

        /* transfer load parameters to DPUs */
        PRINT_MSG("Transfer load parameters to DPUs...");
        DPU_FOREACH(dpu_set, dpu, dpuid) {
            dpu_params[dpuid].kernel = 1;
            dpu_params[dpuid].keys_offs = load_keys_offs;
            dpu_params[dpuid].num_keys = dpu_input_buffers[dpuid]->size();
            if (dpu_params[dpuid].num_keys > max_load_keys_per_dpu) {
                max_load_keys_per_dpu = dpu_params[dpuid].num_keys;
            }
            if (dpu_params[dpuid].num_keys < min_load_keys_per_dpu) {
                min_load_keys_per_dpu = dpu_params[dpuid].num_keys;
            }
            DPU_ASSERT(dpu_prepare_xfer(dpu, &dpu_params[dpuid]));

            // std::cout << dpu_params[dpuid].num_keys << " - " << dpu_input_buffers[dpuid]->capacity() << "\n";
        }
        DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, "dpu_args", 0, sizeof(pimindex_dpu_args_t), DPU_XFER_DEFAULT));

        uint32_t max_load_keys_per_dpu_aligned = (max_load_keys_per_dpu % 2 == 0) ?
                                                 (max_load_keys_per_dpu) :
                                                 (max_load_keys_per_dpu + 1);
        PRINT_MSG("Maximum load keys per DPU: %u", max_load_keys_per_dpu);
        PRINT_MSG("Maximum load size per DPU: %.*f MiB", 2, (max_load_keys_per_dpu * KEY_SIZE)/(float)MiB);
        PRINT_MSG("Minimum load keys per DPU: %u", min_load_keys_per_dpu);
        PRINT_MSG("Minimum load size per DPU: %.*f MiB", 2, (min_load_keys_per_dpu * KEY_SIZE)/(float)MiB);

        /* transfer load keys to DPUs */
        PRINT_MSG("Transfer load keys to DPUs...");
        t.start("CPU-DPU xfer (load keys)");
        DPU_FOREACH(dpu_set, dpu, dpuid) {
            DPU_ASSERT(dpu_prepare_xfer(dpu, dpu_input_buffers[dpuid]->data()));
        }
        DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, load_keys_offs, max_load_keys_per_dpu_aligned * KEY_SIZE, DPU_XFER_DEFAULT));
        t.stop();

        for (uint32_t i = 0; i < NR_DPUS; i++) {
            delete dpu_input_buffers[i];
        }

        /* launch the index load kernel */
        PRINT_MSG("Executing the load kernel on DPUs...");
        t.start("DPU exec (load kernel)");
        DPU_ASSERT(dpu_launch(dpu_set, DPU_SYNCHRONOUS));
        t.stop();

#ifdef PRINT_DPU_LOGS
        /* dump DPU logs */
        PRINT_MSG("Dump DPU logs...");
        DPU_FOREACH(dpu_set, dpu, dpuid) {
            DPU_ASSERT(dpu_log_read(dpu, stdout));
        }
#endif
#endif

#ifdef DATA_FILE
        std::string insert_file(DATA_FILE);
        insert_file += "_insert";
        if (access(insert_file.c_str(), F_OK) == 0) {
            PRINT_MSG("Read insert data file...");
#ifdef LOAD_INDEX
            insert_workload = &((PIMKey_t*) workload)[load_keys];
#else 
            insert_workload = (PIMKey_t*) workload;
#endif
            FILE* fp = fopen(insert_file.c_str(), "r");
            if (!fp) {
                PRINT_ERROR("Cannot open insert file...");
                return;
            }
            uint32_t inum;
            int iret = fscanf(fp, "%u\n", &inum);
            if (inum != insert_keys) {
                PRINT_ERROR("Incorrect insert file...");
                return;
            }
            for (uint32_t i = 0; i < insert_keys; i++) {
                iret = fscanf(fp, "%u ", &insert_workload[i]);
            }
            fclose(fp);
        }
        else {
            PRINT_MSG("Write insert data file...");

#ifdef LOAD_INDEX
            insert_workload = &((PIMKey_t*) workload)[load_keys];
#else 
            insert_workload = (PIMKey_t*) workload;
#endif
            randomize_keys(insert_workload, insert_keys);
            FILE* fp = fopen(insert_file.c_str(), "w");
            if (!fp) {
                PRINT_ERROR("Cannot open insert file...");
                return;
            }
            fprintf(fp, "%u\n", insert_keys);
            for (uint32_t i = 0; i < insert_keys; i++) {
                fprintf(fp, "%u ", insert_workload[i]);
            }
            fclose(fp);
        }
#else

#ifdef LOAD_INDEX
        insert_workload = &((PIMKey_t*) workload)[load_keys];
#else 
        insert_workload = (PIMKey_t*) workload;
#endif
        randomize_keys(insert_workload, insert_keys);
#endif

        uint32_t insert_part_keys_per_dpu = DIVCEIL(insert_keys, NR_DPUS);
        insert_part_keys_per_dpu =
            ((insert_part_keys_per_dpu * (NR_DPUS - 1)) < insert_keys) ?
            (insert_part_keys_per_dpu) : (insert_part_keys_per_dpu - 1);
        uint32_t max_insert_keys_per_dpu = 0;
        uint32_t min_insert_keys_per_dpu = (uint32_t)(-1);
        uint32_t avg_insert_keys_per_dpu = DIVCEIL(insert_keys, NR_DPUS);
        uint32_t avg_size_insert_keys_per_dpu = avg_insert_keys_per_dpu * KEY_SIZE;
        uint32_t insert_keys_offs = MAX_MRAM_SIZE;
        uint32_t est =
            2 /* 4 */ * (insert_part_keys_per_dpu + insert_part_keys_per_dpu / 30);

        PRINT_TOP_RULE
        PRINT_MSG("Insert Keys: %.*f M", 2, insert_keys/(float)1000000);
        PRINT_MSG("Insert size: %.*f MiB", 2, (insert_keys * KEY_SIZE)/(float)MiB);
        PRINT_MSG("Average insert keys per DPU: %u", avg_insert_keys_per_dpu);
        PRINT_MSG("Average insert size per DPU: %.*f MiB", 2,
            (avg_insert_keys_per_dpu * KEY_SIZE)/(float)MiB);

#ifdef PUSH_BACK
        /* preprocess insert keys for each DPU */
        for (uint32_t i = 0; i < NR_DPUS; i++) {
            dpu_input_buffers[i] = new std::vector<PIMKey_t>();
            dpu_input_buffers[i]->reserve(est);
        }

#ifdef SPUSH_BACK
        PRINT_MSG("Single-threaded preprocess insert keys...");
        t.start("CPU exec (preproc. insert keys)");
        for (uint32_t i = 0; i < insert_keys; i++) {
            PIMKey_t key = insert_workload[i];
            // uint32_t dpuid = ((NR_DPUS - 1) & superfast(key));
            uint32_t dpuid = superfast(key) % NR_DPUS;
            dpu_input_buffers[dpuid]->push_back(key);
            dpu_input_buffers[dpuid]->push_back(DEFAULT_VALUE);
        }
        t.stop();
#else
        uint32_t imarker = 0;
        std::thread *cpu_insert_threads[NR_CPU_THREADS];
        uint32_t insert_keys_per_thread = DIVCEIL(insert_keys, NR_CPU_THREADS);
        cpu_thread_arg_t *thread_params =
            reinterpret_cast<cpu_thread_arg_t *>(malloc(NR_CPU_THREADS *
                                                 (sizeof(cpu_thread_arg_t))));
        PRINT_MSG("Multithreaded preprocess insert keys...");
        t.start("CPU exec (preproc. insert keys)");
        for (uint32_t t = 0; t < NR_CPU_THREADS; t++) {
            uint32_t skip = (t == (NR_CPU_THREADS - 1)) ?
                            (insert_keys - t * insert_keys_per_thread) :
                            (insert_keys_per_thread);
            thread_params[t].beg = imarker;
            thread_params[t].end = imarker + skip;
            thread_params[t].insert = true;
            thread_params[t].workload = insert_workload;
            thread_params[t].buffers = &dpu_input_buffers;
            cpu_insert_threads[t] = new std::thread(copy_keys, &thread_params[t]);
            imarker += skip;
        }
        for (uint32_t t = 0; t < NR_CPU_THREADS; t++) {
            cpu_insert_threads[t]->join();
            delete cpu_insert_threads[t];
        }
        t.stop();
#endif

        /* transfer insert parameters to DPUs */
        PRINT_MSG("Transfer insert parameters to DPUs...");
        DPU_FOREACH(dpu_set, dpu, dpuid) {
            dpu_params[dpuid].kernel = 2;
            dpu_params[dpuid].keys_offs = insert_keys_offs;
            dpu_params[dpuid].num_keys = dpu_input_buffers[dpuid]->size() / 2;
            if (dpu_params[dpuid].num_keys > max_insert_keys_per_dpu) {
                max_insert_keys_per_dpu = dpu_params[dpuid].num_keys;
            }
            if (dpu_params[dpuid].num_keys < min_insert_keys_per_dpu) {
                min_insert_keys_per_dpu = dpu_params[dpuid].num_keys;
            }
            DPU_ASSERT(dpu_prepare_xfer(dpu, &dpu_params[dpuid]));
        }
        DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU,
            "dpu_args", 0, sizeof(pimindex_dpu_args_t), DPU_XFER_DEFAULT));

#else
        std::vector<uint32_t> ikey_counter(NR_DPUS, 0);
        for (uint32_t i = 0; i < NR_DPUS; i++) {
            dpu_input_buffers[i] = new std::vector<PIMKey_t>(est);
        }

        /* preprocess insert keys for each DPU */
        PRINT_MSG("Preprocess insert keys...");
        t.start("CPU exec (preproc. insert keys)");
        for (uint32_t i = 0; i < insert_keys; i++) {
            PIMKey_t key = insert_workload[i];
            // uint32_t dpuid = ((NR_DPUS - 1) & superfast(key));
            uint32_t dpuid = superfast(key) % NR_DPUS;
            (*dpu_input_buffers[dpuid])[ikey_counter[dpuid]++] = key;
            (*dpu_input_buffers[dpuid])[ikey_counter[dpuid]++] = DEFAULT_VALUE;
        }
        t.stop();

        /* transfer insert parameters to DPUs */
        PRINT_MSG("Transfer insert parameters to DPUs...");
        DPU_FOREACH(dpu_set, dpu, dpuid) {
            dpu_params[dpuid].kernel = 2;
            dpu_params[dpuid].keys_offs = insert_keys_offs;
            dpu_params[dpuid].num_keys = ikey_counter[dpuid] / 2;
            if (dpu_params[dpuid].num_keys > max_insert_keys_per_dpu) {
                max_insert_keys_per_dpu = dpu_params[dpuid].num_keys;
            }
            if (dpu_params[dpuid].num_keys < min_insert_keys_per_dpu) {
                min_insert_keys_per_dpu = dpu_params[dpuid].num_keys;
            }
            DPU_ASSERT(dpu_prepare_xfer(dpu, &dpu_params[dpuid]));
        }
        DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU,
            "dpu_args", 0, sizeof(pimindex_dpu_args_t), DPU_XFER_DEFAULT));
#endif

        PRINT_MSG("Maximum insert keys per DPU: %u", max_insert_keys_per_dpu);
        PRINT_MSG("Maximum insert size per DPU: %.*f MiB", 2,
            (max_insert_keys_per_dpu * KEY_SIZE)/(float)MiB);
        PRINT_MSG("Minimum insert keys per DPU: %u", min_insert_keys_per_dpu);
        PRINT_MSG("Minimum insert size per DPU: %.*f MiB", 2,
            (min_insert_keys_per_dpu * KEY_SIZE)/(float)MiB);

        /* transfer insert keys to DPUs */
        PRINT_MSG("Transfer insert keys to DPUs...");
        t.start("CPU-DPU xfer (insert keys)");
        DPU_FOREACH(dpu_set, dpu, dpuid) {
            uint32_t num_keys = dpu_params[dpuid].num_keys;
            DPU_ASSERT(dpu_prepare_xfer(dpu, dpu_input_buffers[dpuid]->data()));
        }
        DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU,
                                 DPU_MRAM_HEAP_POINTER_NAME,
                                 insert_keys_offs,
                                 2 * max_insert_keys_per_dpu * KEY_SIZE,
                                 DPU_XFER_DEFAULT));
        t.stop();

        /* launch the insert kernel */
        PRINT_MSG("Executing the insert kernel on DPUs...");
        t.start("DPU exec (insert kernel)");
        DPU_ASSERT(dpu_launch(dpu_set, DPU_SYNCHRONOUS));
        t.stop();

#ifdef PRINT_DPU_LOGS
        /* dump DPU logs */
        PRINT_MSG("Dump DPU logs...");
        DPU_FOREACH(dpu_set, dpu, dpuid) {
            DPU_ASSERT(dpu_log_read(dpu, stdout));
        }
#endif

#ifdef RESULTS_CHECK
        /* transfer insert validation parameters to CPU */
        PRINT_MSG("Transfer insert validation parameters to CPU...");
        DPU_FOREACH(dpu_set, dpu, dpuid) {
            DPU_ASSERT(dpu_prepare_xfer(dpu, &dpu_params[dpuid]));
        }
        DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_FROM_DPU,
            "dpu_args", 0, sizeof(pimindex_dpu_args_t), DPU_XFER_DEFAULT));

        validation = true;
        for (uint32_t dpuid = 0; dpuid < NR_DPUS; dpuid++) {
            if (dpu_params[dpuid].kret != exec_success) {
                validation = false;
                break;
            }
        }
        if (validation) {
            PRINT_INFO("Insertion correct...");
        }
        else {
            PRINT_ERROR("Insertion not correct...");
        }
#endif

        uint32_t search_keys = insert_keys;
        uint32_t max_search_keys_per_dpu = 0;
        uint32_t min_search_keys_per_dpu = (uint32_t)(-1);
        uint32_t avg_search_keys_per_dpu = DIVCEIL(search_keys, NR_DPUS);
        uint32_t avg_size_search_keys_per_dpu = avg_search_keys_per_dpu * KEY_SIZE;
        uint32_t search_keys_offs = MAX_MRAM_SIZE;
        uint32_t search_keys_per_dpu = DIVCEIL(search_keys, NR_DPUS);
        uint32_t search_keys_per_dpu_aligned = (search_keys_per_dpu % 2 == 0) ?
                                (search_keys_per_dpu) : (search_keys_per_dpu + 1);

        PRINT_TOP_RULE
        PRINT_MSG("Search Keys: %.*f M", 2, search_keys/(float)1000000);
        PRINT_MSG("Search size: %.*f MiB", 2, (search_keys * KEY_SIZE)/(float)MiB);
        PRINT_MSG("Average search keys per DPU: %u", avg_search_keys_per_dpu);
        PRINT_MSG("Average search size per DPU: %.*f MiB", 2,
            (avg_search_keys_per_dpu * KEY_SIZE)/(float)MiB);

#ifdef DATA_FILE
        std::string search_file(DATA_FILE);
        search_file += "_search";
        if (access(search_file.c_str(), F_OK) == 0) {
            PRINT_MSG("Read search data file...");

            search_workload = insert_workload;
            FILE* fp = fopen(search_file.c_str(), "r");
            if (!fp) {
                PRINT_ERROR("Cannot open search file...");
                return;
            }
            uint32_t snum;
            int sret = fscanf(fp, "%u\n", &snum);
            if (snum != search_keys) {
                PRINT_ERROR("Incorrect search file...");
                return;
            }
            for (uint32_t i = 0; i < search_keys; i++) {
                sret = fscanf(fp, "%u ", &search_workload[i]);
            }
            fclose(fp);
        }
        else {
            PRINT_MSG("Write search data file...");

            search_workload = insert_workload;
            randomize_keys(search_workload, search_keys);
            FILE* fp = fopen(search_file.c_str(), "w");
            if (!fp) {
                PRINT_ERROR("Cannot open search file...");
                return;
            }
            fprintf(fp, "%u\n", search_keys);
            for (uint32_t i = 0; i < search_keys; i++) {
                fprintf(fp, "%u ", search_workload[i]);
            }
            fclose(fp);
        }
#else
        search_workload = insert_workload;
        randomize_keys(search_workload, search_keys);
#endif

#ifdef PUSH_BACK
        /* preprocess search keys for each DPU */
        for (uint32_t i = 0; i < NR_DPUS; i++) {
            dpu_input_buffers[i]->clear();
            dpu_input_buffers[i]->reserve(est);
        }

#ifdef SPUSH_BACK
        PRINT_MSG("Single-threaded preprocess search keys...");
        t.start("CPU exec (preproc. search keys)");
        for (uint32_t i = 0; i < insert_keys; i++) {
            PIMKey_t key = search_workload[i];
            // uint32_t dpuid = ((NR_DPUS - 1) & superfast(key));
            uint32_t dpuid = superfast(key) % NR_DPUS;
            dpu_input_buffers[dpuid]->push_back(key);
        }
        t.stop();
#else
        uint32_t smarker = 0;
        std::thread *cpu_search_threads[NR_CPU_THREADS];
        uint32_t search_keys_per_thread = DIVCEIL(insert_keys, NR_CPU_THREADS);
        PRINT_MSG("Multithreaded preprocess search keys...");
        t.start("CPU exec (preproc. search keys)");
        for (uint32_t t = 0; t < NR_CPU_THREADS; t++) {
            uint32_t skip = (t == (NR_CPU_THREADS - 1)) ?
                            (insert_keys - t * search_keys_per_thread) :
                            (search_keys_per_thread);
            thread_params[t].beg = smarker;
            thread_params[t].end = smarker + skip;
            thread_params[t].insert = false;
            thread_params[t].workload = search_workload;
            thread_params[t].buffers = &dpu_input_buffers;
            cpu_search_threads[t] = new std::thread(copy_keys, &thread_params[t]);
            smarker += skip;
        }
        for (uint32_t t = 0; t < NR_CPU_THREADS; t++) {
            cpu_search_threads[t]->join();
            delete cpu_search_threads[t];
        }
        t.stop();
        delete thread_params;
#endif

        /* transfer search parameters to DPUs */
        PRINT_MSG("Transfer search parameters to DPUs...");
        DPU_FOREACH(dpu_set, dpu, dpuid) {
            dpu_params[dpuid].kernel = 3;
            dpu_params[dpuid].keys_offs = search_keys_offs;
            dpu_params[dpuid].num_keys = dpu_input_buffers[dpuid]->size();
            if (dpu_params[dpuid].num_keys > max_search_keys_per_dpu) {
                max_search_keys_per_dpu = dpu_params[dpuid].num_keys;
            }
            if (dpu_params[dpuid].num_keys < min_search_keys_per_dpu) {
                min_search_keys_per_dpu = dpu_params[dpuid].num_keys;
            }
            DPU_ASSERT(dpu_prepare_xfer(dpu, &dpu_params[dpuid]));
        }
        DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU,
            "dpu_args", 0, sizeof(pimindex_dpu_args_t), DPU_XFER_DEFAULT));

#else
        std::vector<uint32_t> skey_counter(NR_DPUS, 0);
        for (uint32_t i = 0; i < NR_DPUS; i++) {
            dpu_input_buffers[i] = new std::vector<PIMKey_t>(est);
        }

        /* preprocess search keys for each DPU */
        PRINT_MSG("Preprocess search keys...");
        t.start("CPU exec (preproc. search keys)");
        for (uint32_t i = 0; i < search_keys; i++) {
            PIMKey_t key = search_workload[i];
            // uint32_t dpuid = ((NR_DPUS - 1) & superfast(key));
            uint32_t dpuid = superfast(key) % NR_DPUS;
            (*dpu_input_buffers[dpuid])[skey_counter[dpuid]++] = key;
        }
        t.stop();

        /* transfer search parameters to DPUs */
        PRINT_MSG("Transfer search parameters to DPUs...");
        DPU_FOREACH(dpu_set, dpu, dpuid) {
            dpu_params[dpuid].kernel = 3;
            dpu_params[dpuid].keys_offs = search_keys_offs;
            dpu_params[dpuid].num_keys = skey_counter[dpuid];
            if (dpu_params[dpuid].num_keys > max_search_keys_per_dpu) {
                max_search_keys_per_dpu = dpu_params[dpuid].num_keys;
            }
            if (dpu_params[dpuid].num_keys < min_search_keys_per_dpu) {
                min_search_keys_per_dpu = dpu_params[dpuid].num_keys;
            }
            DPU_ASSERT(dpu_prepare_xfer(dpu, &dpu_params[dpuid]));
        }
        DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU,
            "dpu_args", 0, sizeof(pimindex_dpu_args_t), DPU_XFER_DEFAULT));
#endif

        uint32_t max_search_keys_per_dpu_aligned =
            (max_search_keys_per_dpu % 2 == 0) ?
                (max_search_keys_per_dpu) : (max_search_keys_per_dpu + 1);
        PRINT_MSG("Maximum search keys per DPU: %u", max_search_keys_per_dpu);
        PRINT_MSG("Maximum search size per DPU: %.*f MiB", 2,
            (max_search_keys_per_dpu * KEY_SIZE)/(float)MiB);
        PRINT_MSG("Minimum search keys per DPU: %u", min_search_keys_per_dpu);
        PRINT_MSG("Minimum search size per DPU: %.*f MiB", 2,
            (min_search_keys_per_dpu * KEY_SIZE)/(float)MiB);

        /* transfer search keys to DPUs */
        PRINT_MSG("Transfer search keys to DPUs...");
        t.start("CPU-DPU xfer (search keys)");
        DPU_FOREACH(dpu_set, dpu, dpuid) {
            DPU_ASSERT(dpu_prepare_xfer(dpu, dpu_input_buffers[dpuid]->data()));
        }
        DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU,
                                 DPU_MRAM_HEAP_POINTER_NAME,
                                 search_keys_offs,
                                 max_search_keys_per_dpu_aligned * KEY_SIZE,
                                 DPU_XFER_DEFAULT));
        t.stop();

        /* launch the search kernel */
        PRINT_MSG("Executing the search kernel on DPUs...");
        t.start("DPU exec (search kernel)");
        DPU_ASSERT(dpu_launch(dpu_set, DPU_SYNCHRONOUS));
        t.stop();

#ifdef PRINT_DPU_LOGS
        /* dump DPU logs */
        PRINT_MSG("Dump DPU logs...");
        DPU_FOREACH(dpu_set, dpu, dpuid) {
            DPU_ASSERT(dpu_log_read(dpu, stdout));
        }
#endif

        /* transfer search results to CPU */
        PRINT_MSG("Transfer search results to CPU...");
        t.start("DPU-CPU xfer (search results)");
        DPU_FOREACH(dpu_set, dpu, dpuid) {
            DPU_ASSERT(dpu_prepare_xfer(dpu, dpu_input_buffers[dpuid]->data()));
        }
        DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_FROM_DPU,
                                 DPU_MRAM_HEAP_POINTER_NAME,
                                 search_keys_offs,
                                 max_search_keys_per_dpu_aligned * KEY_SIZE,
                                 DPU_XFER_DEFAULT));
        t.stop();

#ifdef RESULTS_CHECK
        /* validate results */
        validation = true;
        uint32_t found = 0;
        uint32_t not_found = 0;
        for (uint32_t dpuid = 0; dpuid < NR_DPUS; dpuid++) {
            uint32_t vals = dpu_params[dpuid].num_keys;
            for (uint32_t v = 0; v < vals; v++) {
                if ((*dpu_input_buffers[dpuid])[v] == DEFAULT_VALUE) {
                    found++;
                }
                else {
                    not_found++;
                    validation = false;
                }
            }
        }
        if (!validation) {
            PRINT_ERROR("Search not correct...");
        }

        /* transfer search validation parameters to CPU */
        PRINT_MSG("Transfer search validation parameters to CPU...");
        DPU_FOREACH(dpu_set, dpu, dpuid) {
            DPU_ASSERT(dpu_prepare_xfer(dpu, &dpu_params[dpuid]));
        }
        DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_FROM_DPU,
            "dpu_args", 0, sizeof(pimindex_dpu_args_t), DPU_XFER_DEFAULT));

        validation = true;
        for (uint32_t dpuid = 0; dpuid < NR_DPUS; dpuid++) {
            if (dpu_params[dpuid].kret != exec_success) {
                validation = false;
                break;
            }
        }
        PRINT_MSG("Not found: %u", not_found);
        PRINT_MSG("Found: %u", found);
        validation = validation && (found == search_keys);
        if (validation) {
            PRINT_INFO("Search correct...");
        }
        else {
            PRINT_ERROR("Search not correct...");
        }
#endif

        /* free DPUs */
        PRINT_MSG("Free DPUs...");
        DPU_ASSERT(dpu_free(dpu_set));

        delete zeros;
        delete mram_mgr;
        for (uint32_t i = 0; i < NR_DPUS; i++) {
            delete dpu_input_buffers[i];
        }

        // PRINT_TOP_RULE
        // t.print();
        // auto mark = std::to_string(NR_DPUS) + "DPUS_" +
        //             std::to_string(NR_TASKLETS) + "TASKLETS_" +
        //             std::to_string((load_keys/1000000)) + "MLoads" +
        //             std::to_string((search_keys/1000000)) + "MOps";
        // t.print_to_csv(CSV_FILE, mark);
    }
    catch (const dpu::DpuError &e) {
        std::cerr << e.what() << std::endl;
    }
}
