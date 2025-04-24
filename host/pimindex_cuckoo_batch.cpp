#include "pimindex.hpp"

// #define PRINT_DPU_LOGS
#define MAPPING_CHECK


void run_pimindex_cuckoo_batch(void* workload, uint32_t batch_keys, uint32_t insert_keys) {

    try {

        /* allocate DPUs and load binary */
        uint32_t num_dpus, dpuid;
        struct dpu_set_t dpu_set, dpu;

        DPU_ASSERT(dpu_alloc(NR_DPUS, DPU_PROFILE, &dpu_set));
        DPU_ASSERT(dpu_get_nr_dpus(dpu_set, &num_dpus));
        assert(num_dpus == NR_DPUS);
        DPU_ASSERT(dpu_load(dpu_set, INDEX_INSERT_BIN, NULL));

        PRINT_TOP_RULE
        PRINT_INFO("PIM INDEX: Cuckoo Batch");
        PRINT_TOP_RULE
        PRINT_MSG("DPUs: %u", NR_DPUS);
        PRINT_MSG("Tasklets: %u", NR_TASKLETS);
        PRINT_MSG("Index entry size: %lu", BUCKET_ENTRY_SIZE);
        PRINT_MSG("Key size: %lu", KEY_SIZE);
        PRINT_MSG("Bucket size: %lu", BUCKET_SIZE);
        PRINT_MSG("Max. buckets: %lu", MAX_NUM_BUCKETS);
        PRINT_MSG("Max. actual buckets: %u", (1u << MAX_INDEX_EXPONENT));
        PRINT_MSG("Max. bucket locks: %lu", MAX_NUM_BUCKET_LOCKS);
        PRINT_MSG("Max. bucket lock size: %.2f KiB",
                        MAX_SIZE_BUCKET_LOCKS/(float)KiB);
        PRINT_MSG("Bucket locks: %u", NR_BUCKET_LOCKS);
        PRINT_MSG("Bucket lock size: %.2f KiB",
                        (NR_BUCKET_LOCKS * sizeof(BucketLock))/(float)KiB);

        timer t;

        PIMKey_t *batch_workload;
        PIMKey_t *insert_workload = (PIMKey_t*) workload;
        PIMKey_t *search_workload = (PIMKey_t*) workload;

        void** sg_pptrs;
        PIMKey_t*** sg_part_ptrs_raw;
        uint32_t* sg_part_sizes_raw[NR_DPUS];
        get_block_t sg_local_partition;
        sg_partition_xfer_args_t sg_params;

        mram_mem_mgr_t *mram_mgr = new mram_mem_mgr_t();
        std::vector<uint32_t> chunk_sizes(NR_DPUS, CHUNK_SIZE);
        std::vector<mram_heap_obj_t*> chunk_blocks(MAX_NUM_CHUNKS); /* TODO: struct for instances of the same block type */
        std::vector<pimindex_dpu_args_t> dpu_params(NR_DPUS);
        std::vector<std::vector<PIMKey_t>*> dpu_input_buffers(NR_DPUS);
        std::vector<std::vector<PIMKey_t>*> check_dpu_input_buffers(NR_DPUS);
        std::vector<std::vector<uint32_t>*> local_prefix_sums(NR_DPUS);
        std::vector<std::vector<uint32_t>*> global_prefix_sums(NR_DPUS);
        std::vector<std::vector<uint32_t>*> sg_part_sizes(NR_DPUS);
        std::vector<uint32_t> sg_num_parts(NR_DPUS, NR_DPUS);
        std::vector<std::vector<PIMKey_t*>*> sg_part_ptrs(NR_DPUS);
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

        bool validation;
        uint32_t total_padding;
        uint32_t total_part_sizes;
        uint32_t total_buckets;
        uint32_t max_buckets_per_dpu;
        uint32_t min_buckets_per_dpu;
        uint32_t total_keys;
        uint32_t max_keys_per_dpu;
        uint32_t min_keys_per_dpu;
        if (batch_keys == 0) {
            PRINT_ERROR("Batch keys undefined...");
            return;
        }

        uint32_t prefix = 0;
        uint32_t batch_insert_keys = batch_keys;
        uint32_t batch_insert_part_keys_per_dpu =
                                DIVCEIL(batch_insert_keys, NR_DPUS);
        batch_insert_part_keys_per_dpu =
            ((batch_insert_part_keys_per_dpu * (NR_DPUS - 1)) < batch_insert_keys) ?
            (batch_insert_part_keys_per_dpu) : (batch_insert_part_keys_per_dpu - 1);
        uint32_t size_batch_insert_part_keys_per_dpu =
                                batch_insert_part_keys_per_dpu * KEY_SIZE;
        uint32_t batch_insert_part_keys_offs = 2 * MAX_INDEX_SPACE + (2 * MiB);
        uint32_t max_batch_insert_keys_per_dpu = 0;
        uint32_t min_batch_insert_keys_per_dpu = (uint32_t)(-1);
        uint32_t avg_batch_insert_keys_per_dpu = DIVCEIL(batch_insert_keys, NR_DPUS);
        uint32_t avg_size_batch_insert_keys_per_dpu =
                                avg_batch_insert_keys_per_dpu * KEY_SIZE;
        uint32_t batch_insert_keys_offs = 2 * MAX_INDEX_SPACE + (2 * MiB);
        uint32_t est = 2 /* 4 */ * (batch_insert_part_keys_per_dpu +
                                    batch_insert_part_keys_per_dpu / 30);
        uint32_t num_batches = insert_keys / batch_insert_keys;
        assert(insert_keys % batch_insert_keys  == 0);
        srand(time(nullptr));

        for (uint32_t i = 0; i < NR_DPUS; i++) {
            dpu_input_buffers[i] = new std::vector<PIMKey_t>(est);
        }
        for (uint32_t i = 0; i < NR_DPUS; i++) {
            local_prefix_sums[i] = new std::vector<uint32_t>(NR_DPUS);
        }
        for (uint32_t i = 0; i < NR_DPUS; i++) {
            global_prefix_sums[i] = new std::vector<uint32_t>(NR_DPUS);
        }
        for (uint32_t i = 0; i < NR_DPUS; i++) {
            sg_part_sizes[i] = new std::vector<uint32_t>(NR_DPUS);
        }
        for (uint32_t i = 0; i < NR_DPUS; i++) {
            sg_part_ptrs[i] = new std::vector<PIMKey_t*>(NR_DPUS);
        }
        sg_pptrs = (void**) malloc(NR_DPUS * sizeof(void*));
        sg_part_ptrs_raw = (PIMKey_t***) sg_pptrs;

        PRINT_TOP_RULE
        PRINT_MSG("est: %u", est);
        PRINT_MSG("Batches: %u", num_batches);
        PRINT_MSG("Batch insert mapping keys: %.*f M", 2,
                        batch_insert_keys/(float)1000000);
        PRINT_MSG("Batch insert mapping size: %.*f MiB", 2,
                        (batch_insert_keys * KEY_SIZE)/(float)MiB);
        PRINT_MSG("Batch insert mapping keys per DPU: %u",
                        batch_insert_part_keys_per_dpu);
        PRINT_MSG("Size of batch insert mapping keys per DPU: %.*f MiB", 2,
                        (batch_insert_part_keys_per_dpu * KEY_SIZE)/(float)MiB);

#ifdef DATA_FILE
        std::string insert_file(DATA_FILE);
        insert_file += "_insert";
        if (access(insert_file.c_str(), F_OK) == 0) {
            PRINT_MSG("Read insert data file...");
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
        randomize_keys(insert_workload, insert_keys);
#endif

        for (uint32_t batch = 0; batch < num_batches; batch++) {
            PRINT_TOP_RULE
            PRINT_MSG("Insert Batch %u...", (batch + 1));

#ifdef MEM_UTILIZATION
            total_buckets = 0;
            max_buckets_per_dpu = 0;
            min_buckets_per_dpu = (uint32_t)(-1);
            total_keys = 0;
            max_keys_per_dpu = 0;
            min_keys_per_dpu = (uint32_t)(-1);

            /* transfer memory utilization parameters to DPUs */
            PRINT_MSG("Transfer memory utilization parameters to DPUs...");
            DPU_FOREACH(dpu_set, dpu, dpuid) {
                dpu_params[dpuid].kernel = 5;
                DPU_ASSERT(dpu_prepare_xfer(dpu, &dpu_params[dpuid]));
            }
            DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU,
                "dpu_args", 0, sizeof(pimindex_dpu_args_t), DPU_XFER_DEFAULT));

            /* launch the memory utilization kernel */
            PRINT_MSG("Executing the memory utilization kernel on DPUs...");
            DPU_ASSERT(dpu_launch(dpu_set, DPU_SYNCHRONOUS));

            /* transfer memory utilization results to CPU */
            PRINT_MSG("Transfer memory utilization results to CPU...");
            DPU_FOREACH(dpu_set, dpu, dpuid) {
                DPU_ASSERT(dpu_prepare_xfer(dpu, &dpu_params[dpuid]));
            }
            DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_FROM_DPU,
                "dpu_args", 0, sizeof(pimindex_dpu_args_t), DPU_XFER_DEFAULT));

            for (uint32_t dpuid = 0; dpuid < NR_DPUS; dpuid++) {
                uint32_t buckets = dpu_params[dpuid].num_buckets;
                uint32_t keys = dpu_params[dpuid].num_keys;
                total_buckets += buckets;
                total_keys += keys;
                if (buckets > max_buckets_per_dpu) {
                    max_buckets_per_dpu = buckets;
                }
                if (buckets < min_buckets_per_dpu) {
                    min_buckets_per_dpu = buckets;
                }
                if (keys > max_keys_per_dpu) {
                    max_keys_per_dpu = keys;
                }
                if (keys < min_keys_per_dpu) {
                    min_keys_per_dpu = keys;
                }
            }

            PRINT_MSG("Total buckets: %u", total_buckets);
            PRINT_MSG("Maximum buckets per DPU: %u", max_buckets_per_dpu);
            PRINT_MSG("Minimum buckets per DPU: %u", min_buckets_per_dpu);
            PRINT_MSG("Total keys: %u", total_keys);
            PRINT_MSG("Maximum keys per DPU: %u", max_keys_per_dpu);
            PRINT_MSG("Minimum keys per DPU: %u", min_keys_per_dpu);
            PRINT_MSG("Maximum memory utilization per DPU: %.*f", 2,
                            (max_keys_per_dpu)/(float)(min_buckets_per_dpu *
                                    ENTRIES_PER_BUCKET_PIM)); /* TODO: update */
            PRINT_MSG("Minimum memory utilization per DPU: %.*f", 2,
                            (min_keys_per_dpu)/(float)(max_buckets_per_dpu *
                                    ENTRIES_PER_BUCKET_PIM)); /* TODO: update */
#endif

            batch_workload = &insert_workload[batch * batch_insert_keys];
            for (uint32_t i = 0; i < NR_DPUS; i++) {
                uint32_t keys =
                    (i == (NR_DPUS - 1)) ?
                    (batch_insert_keys - i * batch_insert_part_keys_per_dpu) :
                    (batch_insert_part_keys_per_dpu);
                dpu_params[i].num_keys = keys;
                for (uint32_t k = 0; k < keys; k++) {
                    PIMKey_t key =
                        batch_workload[i * batch_insert_part_keys_per_dpu + k];
                    (*dpu_input_buffers[i])[2 * k] = key;
                    (*dpu_input_buffers[i])[2 * k + 1] = DEFAULT_VALUE;
                }
            }

            /* transfer batch insert mapping parameters to DPUs */
            PRINT_MSG("Transfer batch insert mapping parameters to DPUs...");
            DPU_FOREACH(dpu_set, dpu, dpuid) {
                dpu_params[dpuid].kernel = 4;
                dpu_params[dpuid].keys_offs = batch_insert_part_keys_offs;
                DPU_ASSERT(dpu_prepare_xfer(dpu, &dpu_params[dpuid]));
            }
            DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU,
                "dpu_args", 0, sizeof(pimindex_dpu_args_t), DPU_XFER_DEFAULT));

            /* transfer batch insert mapping keys to DPUs */
            PRINT_MSG("Transfer batch insert mapping keys to DPUs...");
            t.start("Batch: " + std::to_string(batch + 1) +
                    " | CPU-DPU xfer (batch insert mapping keys)");
            DPU_FOREACH(dpu_set, dpu, dpuid) {
                DPU_ASSERT(dpu_prepare_xfer(dpu,
                                            dpu_input_buffers[dpuid]->data()));
            }
            DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU,
                                     DPU_MRAM_HEAP_POINTER_NAME,
                                     batch_insert_part_keys_offs,
                                     2 * batch_insert_part_keys_per_dpu * KEY_SIZE,
                                     DPU_XFER_DEFAULT));
            t.stop();

            /* launch the mapping kernel */
            PRINT_MSG("Executing the insert mapping kernel on DPUs...");
            t.start("Batch: " + std::to_string(batch + 1) +
                    " | DPU exec (insert mapping kernel)");
            DPU_ASSERT(dpu_launch(dpu_set, DPU_SYNCHRONOUS));
            t.stop();

#ifdef PRINT_DPU_LOGS
            /* dump DPU logs */
            PRINT_MSG("Dump DPU logs...");
            DPU_FOREACH(dpu_set, dpu, dpuid) {
                DPU_ASSERT(dpu_log_read(dpu, stdout));
        }
#endif

#ifdef MAPPING_CHECK
            /* transfer mapping validation parameters to CPU */
            PRINT_MSG("Transfer mapping validation parameters to CPU...");
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
            if (!validation) {
                PRINT_ERROR("Insert mapping not valid...");
            }
#endif

            /* transfer DPU histograms to CPU */
            PRINT_MSG("Transfer DPU histograms to CPU...");
            t.start("Batch: " + std::to_string(batch + 1) +
                    " | DPU-CPU xfer (batch insert mapping histograms)");
            DPU_FOREACH(dpu_set, dpu, dpuid) {
                DPU_ASSERT(dpu_prepare_xfer(dpu,
                                            local_prefix_sums[dpuid]->data()));
            }
            DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_FROM_DPU,
                                     DPU_MRAM_HEAP_POINTER_NAME,
                                     batch_insert_part_keys_offs,
                                     NR_DPUS * sizeof(uint32_t),
                                     DPU_XFER_DEFAULT));
            t.stop();

            for (uint32_t dpuid = 0; dpuid < NR_DPUS; dpuid++) {
                std::vector<uint32_t>& prefixes = (*local_prefix_sums[dpuid]);
                std::vector<uint32_t>& sizes = (*sg_part_sizes[dpuid]);
                for (uint32_t part = (NR_DPUS - 1); part > 0; part--) {
                    uint32_t keys = prefixes[part] - prefixes[part - 1];
                    sizes[part] = 2 * keys;
                    prefixes[part] -= keys;
                }
                sizes[0] = 2 * prefixes[0];
                prefixes[0] = 0;
            }
            for (uint32_t part = 0; part < NR_DPUS; part++) {
                prefix = 0;
                for (uint32_t dpuid = 0; dpuid < NR_DPUS; dpuid++) {
                    (*global_prefix_sums[part])[dpuid] = prefix;
                    prefix += (*sg_part_sizes[dpuid])[part];
                }
                dpu_params[part].num_keys = prefix / 2;
                if (dpu_params[part].num_keys > max_batch_insert_keys_per_dpu) {
                    max_batch_insert_keys_per_dpu = dpu_params[part].num_keys;
                }
                if (dpu_params[part].num_keys < min_batch_insert_keys_per_dpu) {
                    min_batch_insert_keys_per_dpu = dpu_params[part].num_keys;
                }
            }
            for (uint32_t dpuid = 0; dpuid < NR_DPUS; dpuid++) {
                for (uint32_t part = 0; part < NR_DPUS; part++) {
                    uint32_t offs = (*global_prefix_sums[part])[dpuid];
                    (*sg_part_ptrs[dpuid])[part] =
                                            &(*dpu_input_buffers[part])[offs];
                }
            }
            for (uint32_t dpuid = 0; dpuid < NR_DPUS; dpuid++) {
                sg_part_sizes_raw[dpuid] = sg_part_sizes[dpuid]->data();
            }
            for (uint32_t dpuid = 0; dpuid < NR_DPUS; dpuid++) {
                sg_part_ptrs_raw[dpuid] = sg_part_ptrs[dpuid]->data();
            }
            sg_params.num_partitions = sg_num_parts.data();
            sg_params.partition_sizes = sg_part_sizes_raw;
            sg_params.partition_ptrs = sg_part_ptrs_raw;
            sg_local_partition.f = &sg_partition_func;
            sg_local_partition.args = &sg_params;
            sg_local_partition.args_size = sizeof(sg_partition_xfer_args_t);

            /* transfer batch partitions from DPUs to CPU */
            PRINT_MSG("Transfer batch DPU partitions to CPU...");
            t.start("Batch: " + std::to_string(batch + 1) +
                    " | DPU-CPU xfer (batch insert sg partitions)");
            DPU_ASSERT(dpu_push_sg_xfer(dpu_set, DPU_XFER_FROM_DPU,
                                        DPU_MRAM_HEAP_POINTER_NAME,
                                        (batch_insert_part_keys_offs + 6 * MiB),
                                        2 * max_batch_insert_keys_per_dpu * KEY_SIZE,
                                        &sg_local_partition,
                                        DPU_SG_XFER_DISABLE_LENGTH_CHECK));
            t.stop();

#ifdef MAPPING_CHECK
            validation = false;
            total_padding = 0;
            total_part_sizes = 0;
            for (uint32_t part = 0; part < NR_DPUS; part++) {
                auto &vec = (*dpu_input_buffers[part]);
                uint32_t psize = dpu_params[part].num_keys;
                total_part_sizes += psize;
                for (uint32_t i = 0; i < psize; i++) {
                    uint32_t val = vec[i];
                    if (val == 0) {
                        total_padding++;
                    }
                }
            }
            PRINT_MSG("Padding keys: %u", total_padding);
            PRINT_MSG("Partition keys: %u", total_part_sizes);
            validation = (total_padding == 0 && total_part_sizes == batch_insert_keys);
            if (validation) {
                PRINT_INFO("Insert mapping valid...");
            }
            else {
                PRINT_ERROR("Insert mapping not valid...");
            }
#endif

            PRINT_MSG("Batch insert Keys: %.*f M", 2,
                            batch_insert_keys/(float)1000000);
            PRINT_MSG("Batch insert size: %.*f MiB", 2,
                            (batch_insert_keys * KEY_SIZE)/(float)MiB);
            PRINT_MSG("Average batch insert keys per DPU: %u",
                            avg_batch_insert_keys_per_dpu);
            PRINT_MSG("Average batch insert size per DPU: %.*f MiB", 2,
                            (avg_batch_insert_keys_per_dpu * KEY_SIZE)/(float)MiB);
            PRINT_MSG("Maximum batch insert keys per DPU: %u",
                            max_batch_insert_keys_per_dpu);
            PRINT_MSG("Maximum batch insert size per DPU: %.*f MiB", 2,
                            (max_batch_insert_keys_per_dpu * KEY_SIZE)/(float)MiB);
            PRINT_MSG("Minimum batch insert keys per DPU: %u",
                            min_batch_insert_keys_per_dpu);
            PRINT_MSG("Minimum batch insert size per DPU: %.*f MiB", 2,
                            (min_batch_insert_keys_per_dpu * KEY_SIZE)/(float)MiB);

            /* transfer batch insert parameters to DPUs */
            PRINT_MSG("Transfer batch insert parameters to DPUs...");
            DPU_FOREACH(dpu_set, dpu, dpuid) {
                dpu_params[dpuid].kernel = 1;
                dpu_params[dpuid].keys_offs = batch_insert_keys_offs;
                DPU_ASSERT(dpu_prepare_xfer(dpu, &dpu_params[dpuid]));
            }
            DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU,
                "dpu_args", 0, sizeof(pimindex_dpu_args_t), DPU_XFER_DEFAULT));

            /* transfer batch insert keys to DPUs */
            PRINT_MSG("Transfer batch insert keys to DPUs...");
            t.start("Batch: " + std::to_string(batch + 1) +
                    " | CPU-DPU xfer (batch insert keys)");
            DPU_FOREACH(dpu_set, dpu, dpuid) {
                DPU_ASSERT(dpu_prepare_xfer(dpu,
                                            dpu_input_buffers[dpuid]->data()));
            }
            DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU,
                                     DPU_MRAM_HEAP_POINTER_NAME,
                                     batch_insert_keys_offs,
                                     2 * max_batch_insert_keys_per_dpu * KEY_SIZE,
                                     DPU_XFER_DEFAULT));
            t.stop();

            /* launch the insert kernel */
            PRINT_MSG("Executing the insert kernel on DPUs...");
            t.start("Batch: " + std::to_string(batch + 1) +
                    " | DPU exec (insert kernel)");
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
        }

        prefix = 0;
        uint32_t search_keys = insert_keys;
        uint32_t batch_search_keys = batch_insert_keys;
        uint32_t batch_search_part_keys_per_dpu =
                                DIVCEIL(batch_search_keys, NR_DPUS);
        batch_search_part_keys_per_dpu =
            ((batch_search_part_keys_per_dpu * (NR_DPUS - 1)) < batch_search_keys) ?
            (batch_search_part_keys_per_dpu) : (batch_search_part_keys_per_dpu - 1);
        uint32_t size_batch_search_part_keys_per_dpu =
                                batch_search_part_keys_per_dpu * KEY_SIZE;
        uint32_t batch_search_part_keys_offs = 2 * MAX_INDEX_SPACE + (2 * MiB);
        uint32_t max_batch_search_keys_per_dpu = 0;
        uint32_t min_batch_search_keys_per_dpu = (uint32_t)(-1);
        uint32_t avg_batch_search_keys_per_dpu = DIVCEIL(batch_search_keys, NR_DPUS);
        uint32_t avg_size_batch_search_keys_per_dpu =
                                avg_batch_search_keys_per_dpu * KEY_SIZE;
        uint32_t batch_search_keys_offs = 2 * MAX_INDEX_SPACE + (2 * MiB);

        PRINT_TOP_RULE
        PRINT_MSG("Batches: %u", num_batches);
        PRINT_MSG("Batch search mapping keys: %.*f M", 2,
                        batch_search_keys/(float)1000000);
        PRINT_MSG("Batch search mapping size: %.*f MiB", 2,
                        (batch_search_keys * KEY_SIZE)/(float)MiB);
        PRINT_MSG("Batch search mapping keys per DPU: %u",
                        batch_search_part_keys_per_dpu);
        PRINT_MSG("Size of batch search mapping keys per DPU: %.*f MiB", 2,
                        (batch_search_part_keys_per_dpu * KEY_SIZE)/(float)MiB);

#ifdef DATA_FILE
        std::string search_file(DATA_FILE);
        search_file += "_search";
        if (access(search_file.c_str(), F_OK) == 0) {
            PRINT_MSG("Read search data file...");
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
        randomize_keys(search_workload, search_keys);
#endif

        for (uint32_t batch = 0; batch < num_batches; batch++) {
            PRINT_TOP_RULE
            PRINT_MSG("Search Batch %u...", (batch + 1));

#ifdef MEM_UTILIZATION
            total_buckets = 0;
            max_buckets_per_dpu = 0;
            min_buckets_per_dpu = (uint32_t)(-1);
            total_keys = 0;
            max_keys_per_dpu = 0;
            min_keys_per_dpu = (uint32_t)(-1);

            /* transfer memory utilization parameters to DPUs */
            PRINT_MSG("Transfer memory utilization parameters to DPUs...");
            DPU_FOREACH(dpu_set, dpu, dpuid) {
                dpu_params[dpuid].kernel = 5;
                DPU_ASSERT(dpu_prepare_xfer(dpu, &dpu_params[dpuid]));
            }
            DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU,
                "dpu_args", 0, sizeof(pimindex_dpu_args_t), DPU_XFER_DEFAULT));

            /* launch the memory utilization kernel */
            PRINT_MSG("Executing the memory utilization kernel on DPUs...");
            DPU_ASSERT(dpu_launch(dpu_set, DPU_SYNCHRONOUS));

            /* transfer memory utilization results to CPU */
            PRINT_MSG("Transfer memory utilization results to CPU...");
            DPU_FOREACH(dpu_set, dpu, dpuid) {
                DPU_ASSERT(dpu_prepare_xfer(dpu, &dpu_params[dpuid]));
            }
            DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_FROM_DPU,
                "dpu_args", 0, sizeof(pimindex_dpu_args_t), DPU_XFER_DEFAULT));

            for (uint32_t dpuid = 0; dpuid < NR_DPUS; dpuid++) {
                uint32_t buckets = dpu_params[dpuid].num_buckets;
                uint32_t keys = dpu_params[dpuid].num_keys;
                total_buckets += buckets;
                total_keys += keys;
                if (buckets > max_buckets_per_dpu) {
                    max_buckets_per_dpu = buckets;
                }
                if (buckets < min_buckets_per_dpu) {
                    min_buckets_per_dpu = buckets;
                }
                if (keys > max_keys_per_dpu) {
                    max_keys_per_dpu = keys;
                }
                if (keys < min_keys_per_dpu) {
                    min_keys_per_dpu = keys;
                }
            }

            PRINT_MSG("Total buckets: %u", total_buckets);
            PRINT_MSG("Maximum buckets per DPU: %u", max_buckets_per_dpu);
            PRINT_MSG("Minimum buckets per DPU: %u", min_buckets_per_dpu);
            PRINT_MSG("Total keys: %u", total_keys);
            PRINT_MSG("Maximum keys per DPU: %u", max_keys_per_dpu);
            PRINT_MSG("Minimum keys per DPU: %u", min_keys_per_dpu);
            PRINT_MSG("Maximum memory utilization per DPU: %.*f", 2,
                        (max_keys_per_dpu)/(float)(min_buckets_per_dpu *
                                ENTRIES_PER_BUCKET_PIM)); /* TODO: update */
            PRINT_MSG("Minimum memory utilization per DPU: %.*f", 2,
                        (min_keys_per_dpu)/(float)(max_buckets_per_dpu *
                                ENTRIES_PER_BUCKET_PIM)); /* TODO: update */
#endif

            batch_workload = &search_workload[batch * batch_search_keys];
            for (uint32_t i = 0; i < NR_DPUS; i++) {
                uint32_t keys =
                    (i == (NR_DPUS - 1)) ?
                    (batch_search_keys - i * batch_search_part_keys_per_dpu) :
                    (batch_search_part_keys_per_dpu);
                dpu_params[i].num_keys = keys;
                std::memcpy(dpu_input_buffers[i]->data(),
                            &batch_workload[i * batch_search_part_keys_per_dpu],
                            KEY_SIZE * keys);
            }

            /* transfer batch search mapping parameters to DPUs */
            PRINT_MSG("Transfer batch search mapping parameters to DPUs...");
            DPU_FOREACH(dpu_set, dpu, dpuid) {
                dpu_params[dpuid].kernel = 3;
                dpu_params[dpuid].keys_offs = batch_search_part_keys_offs;
                DPU_ASSERT(dpu_prepare_xfer(dpu, &dpu_params[dpuid]));
            }
            DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU,
                "dpu_args", 0, sizeof(pimindex_dpu_args_t), DPU_XFER_DEFAULT));

            uint32_t batch_search_part_keys_per_dpu_aligned =
                                    (batch_search_part_keys_per_dpu % 2 == 0) ?
                                    (batch_search_part_keys_per_dpu) :
                                    (batch_search_part_keys_per_dpu + 1);
            /* transfer batch search mapping keys to DPUs */
            PRINT_MSG("Transfer batch search mapping keys to DPUs...");
            t.start("Batch: " + std::to_string(batch + 1) +
                    " | CPU-DPU xfer (batch search mapping keys)");
            DPU_FOREACH(dpu_set, dpu, dpuid) {
                DPU_ASSERT(dpu_prepare_xfer(dpu, dpu_input_buffers[dpuid]->data()));
            }
            DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU,
                            DPU_MRAM_HEAP_POINTER_NAME,
                            batch_search_part_keys_offs,
                            batch_search_part_keys_per_dpu_aligned * KEY_SIZE,
                            DPU_XFER_DEFAULT));
            t.stop();

            /* launch the mapping kernel */
            PRINT_MSG("Executing the search mapping kernel on DPUs...");
            t.start("Batch: " + std::to_string(batch + 1) +
                    " | DPU exec (search mapping kernel)");
            DPU_ASSERT(dpu_launch(dpu_set, DPU_SYNCHRONOUS));
            t.stop();

#ifdef PRINT_DPU_LOGS
            /* dump DPU logs */
            PRINT_MSG("Dump DPU logs...");
            DPU_FOREACH(dpu_set, dpu, dpuid) {
                DPU_ASSERT(dpu_log_read(dpu, stdout));
            }
#endif

#ifdef MAPPING_CHECK
            /* transfer mapping validation parameters to CPU */
            PRINT_MSG("Transfer mapping validation parameters to CPU...");
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
            if (!validation) {
                PRINT_ERROR("Search mapping not valid...");
            }
#endif

            PRINT_MSG("Transfer DPU histograms to CPU...");
            t.start("Batch: " + std::to_string(batch + 1) +
                    " | DPU-CPU xfer (batch search mapping histograms)");
            DPU_FOREACH(dpu_set, dpu, dpuid) {
                DPU_ASSERT(dpu_prepare_xfer(dpu, local_prefix_sums[dpuid]->data()));
            }
            DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_FROM_DPU,
                                    DPU_MRAM_HEAP_POINTER_NAME,
                                    batch_search_part_keys_offs,
                                    NR_DPUS * sizeof(uint32_t),
                                    DPU_XFER_DEFAULT));
            t.stop();

            for (uint32_t dpuid = 0; dpuid < NR_DPUS; dpuid++) {
                std::vector<uint32_t>& prefixes = (*local_prefix_sums[dpuid]);
                std::vector<uint32_t>& sizes = (*sg_part_sizes[dpuid]);
                for (uint32_t part = (NR_DPUS - 1); part > 0; part--) {
                    uint32_t keys = prefixes[part] - prefixes[part - 1];
                    sizes[part] = keys;
                    prefixes[part] -= keys;
                }
                sizes[0] = prefixes[0];
                prefixes[0] = 0;
            }
            for (uint32_t part = 0; part < NR_DPUS; part++) {
                prefix = 0;
                for (uint32_t dpuid = 0; dpuid < NR_DPUS; dpuid++) {
                    (*global_prefix_sums[part])[dpuid] = prefix;
                    prefix += (*sg_part_sizes[dpuid])[part];
                }
                dpu_params[part].num_keys = prefix;
                if (prefix > max_batch_search_keys_per_dpu) {
                    max_batch_search_keys_per_dpu = prefix;
                }
                if (prefix < min_batch_search_keys_per_dpu) {
                    min_batch_search_keys_per_dpu = prefix;
                }
            }
            for (uint32_t dpuid = 0; dpuid < NR_DPUS; dpuid++) {
                for (uint32_t part = 0; part < NR_DPUS; part++) {
                    uint32_t offs = (*global_prefix_sums[part])[dpuid];
                    (*sg_part_ptrs[dpuid])[part] = &(*dpu_input_buffers[part])[offs];
                }
            }
            for (uint32_t dpuid = 0; dpuid < NR_DPUS; dpuid++) {
                sg_part_sizes_raw[dpuid] = sg_part_sizes[dpuid]->data();
            }
            for (uint32_t dpuid = 0; dpuid < NR_DPUS; dpuid++) {
                sg_part_ptrs_raw[dpuid] = sg_part_ptrs[dpuid]->data();
            }
            sg_params.num_partitions = sg_num_parts.data();
            sg_params.partition_sizes = sg_part_sizes_raw;
            sg_params.partition_ptrs = sg_part_ptrs_raw;
            sg_local_partition.f = &sg_partition_func;
            sg_local_partition.args = &sg_params;
            sg_local_partition.args_size = sizeof(sg_partition_xfer_args_t);
            uint32_t max_batch_search_keys_per_dpu_aligned =
                                    (max_batch_search_keys_per_dpu % 2 == 0) ?
                                    (max_batch_search_keys_per_dpu) :
                                    (max_batch_search_keys_per_dpu + 1);

            /* transfer batch partitions from DPUs to CPU */
            PRINT_MSG("Transfer batch DPU partitions to CPU...");
            t.start("Batch: " + std::to_string(batch + 1) +
                    " | DPU-CPU xfer (batch search sg partitions)");
            DPU_ASSERT(dpu_push_sg_xfer(dpu_set, DPU_XFER_FROM_DPU,
                                DPU_MRAM_HEAP_POINTER_NAME,
                                (batch_search_part_keys_offs + 6 * MiB),
                                max_batch_search_keys_per_dpu_aligned * KEY_SIZE,
                                &sg_local_partition,
                                DPU_SG_XFER_DISABLE_LENGTH_CHECK));
            t.stop();

#ifdef MAPPING_CHECK
            validation = false;
            total_padding = 0;
            total_part_sizes = 0;
            for (uint32_t part = 0; part < NR_DPUS; part++) {
                auto &vec = (*dpu_input_buffers[part]);
                uint32_t psize = dpu_params[part].num_keys;
                total_part_sizes += psize;
                for (uint32_t i = 0; i < psize; i++) {
                    uint32_t val = vec[i];
                    if (val == 0) {
                        total_padding++;
                    }
                }
            }
            PRINT_MSG("Padding keys: %u", total_padding);
            PRINT_MSG("Partition keys: %u", total_part_sizes);
            validation = ((total_padding + batch_search_keys) == total_part_sizes);
            if (validation) {
                PRINT_INFO("Search mapping valid...");
            }
            else {
                PRINT_ERROR("Search mapping not valid...");
            }
#endif

            PRINT_TOP_RULE
            PRINT_MSG("Batch Search Keys: %.*f M", 2,
                            batch_search_keys/(float)1000000);
            PRINT_MSG("Batch search size: %.*f MiB", 2,
                            (batch_search_keys * KEY_SIZE)/(float)MiB);
            PRINT_MSG("Average batch search keys per DPU: %u",
                            avg_batch_search_keys_per_dpu);
            PRINT_MSG("Average batch search size per DPU: %.*f MiB", 2,
                            (avg_batch_search_keys_per_dpu * KEY_SIZE)/(float)MiB);
            PRINT_MSG("Maximum batch search keys per DPU: %u",
                            max_batch_search_keys_per_dpu);
            PRINT_MSG("Maximum batch search size per DPU: %.*f MiB", 2,
                            (max_batch_search_keys_per_dpu * KEY_SIZE)/(float)MiB);
            PRINT_MSG("Minimum batch search keys per DPU: %u",
                            min_batch_search_keys_per_dpu);
            PRINT_MSG("Minimum batch search size per DPU: %.*f MiB", 2,
                            (min_batch_search_keys_per_dpu * KEY_SIZE)/(float)MiB);

            /* transfer batch search parameters to DPUs */
            PRINT_MSG("Transfer batch search parameters to DPUs...");
            DPU_FOREACH(dpu_set, dpu, dpuid) {
                dpu_params[dpuid].kernel = 2;
                dpu_params[dpuid].keys_offs = batch_search_keys_offs;
                DPU_ASSERT(dpu_prepare_xfer(dpu, &dpu_params[dpuid]));
            }
            DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU,
                "dpu_args", 0, sizeof(pimindex_dpu_args_t), DPU_XFER_DEFAULT));

            /* transfer batch search keys to DPUs */
            PRINT_MSG("Transfer batch search keys to DPUs...");
            t.start("Batch: " + std::to_string(batch + 1) +
                    " | CPU-DPU xfer (batch search keys)");
            DPU_FOREACH(dpu_set, dpu, dpuid) {
                DPU_ASSERT(dpu_prepare_xfer(dpu, dpu_input_buffers[dpuid]->data()));
            }
            DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU,
                                    DPU_MRAM_HEAP_POINTER_NAME,
                                    batch_search_keys_offs,
                                    max_batch_search_keys_per_dpu_aligned * KEY_SIZE,
                                    DPU_XFER_DEFAULT));
            t.stop();

            /* launch the search kernel */
            PRINT_MSG("Executing the search kernel on DPUs...");
            t.start("Batch: " + std::to_string(batch + 1) +
                    " | DPU exec (search kernel)");
            DPU_ASSERT(dpu_launch(dpu_set, DPU_SYNCHRONOUS));
            t.stop();

#ifdef PRINT_DPU_LOGS
            /* dump DPU logs */
            PRINT_MSG("Dump DPU logs...");
            DPU_FOREACH(dpu_set, dpu, dpuid) {
                DPU_ASSERT(dpu_log_read(dpu, stdout));
            }
#endif

            /* transfer batch search results to CPU */
            PRINT_MSG("Transfer batch search results to CPU...");
            t.start("Batch: " + std::to_string(batch + 1) +
                    " | DPU-CPU xfer (batch search results)");
            DPU_FOREACH(dpu_set, dpu, dpuid) {
                DPU_ASSERT(dpu_prepare_xfer(dpu, dpu_input_buffers[dpuid]->data()));
            }
            DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_FROM_DPU,
                                    DPU_MRAM_HEAP_POINTER_NAME,
                                    batch_search_keys_offs,
                                    max_batch_search_keys_per_dpu_aligned * KEY_SIZE,
                                    DPU_XFER_DEFAULT));
            t.stop();

#ifdef RESULTS_CHECK
            /* validate results */
            validation = false;
            uint32_t found = 0;
            uint32_t padding = 0;
            for (uint32_t dpuid = 0; dpuid < NR_DPUS; dpuid++) {
                uint32_t vals = dpu_params[dpuid].num_keys;
                for (uint32_t v = 0; v < vals; v++) {
                    if ((*dpu_input_buffers[dpuid])[v] == DEFAULT_VALUE) {
                        found++;
                    }
                    else {
                        padding++;
                    }
                }
            }
            validation = (padding == total_padding);
            if (!validation) {
                PRINT_ERROR("Padding not correct...");
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
            PRINT_MSG("Found: %u", found);
            validation = validation && (found == batch_search_keys);
            if (validation) {
                PRINT_INFO("Search correct...");
            }
            else {
                PRINT_ERROR("Search not correct...");
            }
#endif
        }

        /* free DPUs */
        PRINT_MSG("Free DPUs...");
        DPU_ASSERT(dpu_free(dpu_set));

        delete mram_mgr;
        for (uint32_t i = 0; i < NR_DPUS; i++) {
            delete dpu_input_buffers[i];
        }
        for (uint32_t i = 0; i < NR_DPUS; i++) {
            delete local_prefix_sums[i];
        }
        for (uint32_t i = 0; i < NR_DPUS; i++) {
            delete global_prefix_sums[i];
        }
        for (uint32_t i = 0; i < NR_DPUS; i++) {
            delete sg_part_sizes[i];
        }
        for (uint32_t i = 0; i < NR_DPUS; i++) {
            delete sg_part_ptrs[i];
        }

        // PRINT_TOP_RULE
        // t.print();
        // auto mark = std::to_string(NR_DPUS) + "DPUS_" +
        //             std::to_string(NR_TASKLETS) + "TASKLETS_" +
        //             std::to_string((batch_keys/1000000)) + "MLoads" +
        //             std::to_string((search_keys/1000000)) + "MOps";
        // t.print_to_csv(CSV_FILE, mark);
    }
    catch (const dpu::DpuError &e) {
        std::cerr << e.what() << std::endl;
    }
}
