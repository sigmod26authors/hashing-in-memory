#include "pimindex.hpp"


bool sg_partition_func(struct sg_block_info* cpu_buffer, uint32_t dpuid, uint32_t part, void* params) {
    sg_partition_xfer_args_t* sg_params = (sg_partition_xfer_args_t*) params;
    uint32_t* sg_parts = sg_params->num_partitions;
    uint32_t** sg_part_sizes = sg_params->partition_sizes;
    PIMKey_t*** sg_part_ptrs = sg_params->partition_ptrs;

    if (part >= sg_parts[dpuid]) { /* number of partitions on the DPU exceeded */
        return false;
    }

    cpu_buffer->addr = (uint8_t*) sg_part_ptrs[dpuid][part];
    cpu_buffer->length = sg_part_sizes[dpuid][part] * KEY_SIZE;

    return true;
}

void* multidim_malloc(uint32_t rows, std::vector<std::vector<PIMKey_t*>*>& ptrs) {
    void** matrix = (void**) malloc(rows * sizeof(void*));
    for (uint32_t i = 0; i < rows; i++) {
        matrix[i] = ptrs[i]->data();
    }

    return matrix;
}

void randomize_keys(PIMKey_t* workload, uint32_t len) {
    srand(time(nullptr));
    for (uint32_t i = 0; i < 2 * len; i++) {
        int pos = rand() % len;
        PIMKey_t tmp = workload[pos];
        workload[pos] = workload[0];
        workload[0] = tmp;
    }
}

void* generate_workload(uint32_t num_keys) {
    void* workload = malloc(num_keys * sizeof(uint32_t));
    generate_keys(workload, num_keys);
    return workload;
}

void generate_keys(void *workload, uint32_t num_keys) {
    uint32_t *array = reinterpret_cast<uint32_t *>(workload);
    uint32_t *keys = array + NR_LOADS;
    uint32_t *nkeys = keys + NR_OPERATIONS;

    srand(time(nullptr));

    uint32_t key;
    uint32_t nkey;
    uint32_t skip;
    uint32_t mark = 0;

    for (uint32_t i = 0; i < NR_LOADS; ++i) {
        skip = rand() % 10;
        if (skip > 2) {
            key = mark + skip;
            mark = mark + skip + 2;
        }
        else {
            key = ++mark;
            ++mark;
        }
        array[i] = key;
    }

    for (uint32_t i = 0; i < NR_OPERATIONS; ++i) {
        skip = rand() % 10;
        if (skip > 2) {
            key = mark + skip;
            nkey = mark + skip + 1;
            mark = mark + skip + 2;
        }
        else {
            key = ++mark;
            nkey = ++mark;
            ++mark;
        }
        keys[i] = key;
        nkeys[i] = nkey;
    }
}

uint32_t get_type_size(uint32_t type) {
    /* TODO */
    return sizeof(uint32_t);
}

mram_heap_obj_t::mram_heap_obj_t() {
    block_sizes.resize(NR_DPUS, 0);
}

mram_mem_mgr_t::mram_mem_mgr_t() {
    heap_ = new mram_heap_t();
    offsets_.push_back(0);
    offsets_.push_back((64 * MiB));
    slots_.push_back(std::make_pair(true, nullptr));
}

mram_mem_mgr_t::~mram_mem_mgr_t() {
    /* TODO */
}

mram_heap_obj_t* mram_mem_mgr_t::alloc_block(const std::string &name,
    std::vector<uint32_t>& block_sizes, uint32_t max_block_size,
        uint32_t align_size, uint32_t type) {

    assert(align_size != 0);
    max_block_size += ((8 - (max_block_size % 8)) % 8); /* make block size 8-byte aligned */

    mram_heap_obj_t* new_obj = new mram_heap_obj_t();
    uint32_t offs = this->get_slot(new_obj, max_block_size, align_size);
    heap_->objs.push_back(new_obj);
    auto &obj = heap_->objs.back();

    obj->offset = offs;
    obj->max_block_size = max_block_size; /* TODO: unspecified max. block size */
    obj->elem_type = type;
    obj->elem_size = get_type_size(type);
    obj->name.assign(name);

    assert(block_sizes.size() == NR_DPUS);
    for (auto i = 0; i < NR_DPUS; i++) {
        if (block_sizes[i] > max_block_size) {
            PRINT_ERROR("MRAM block size error");
            std::exit(EXIT_FAILURE);
        }
        else {
            obj->block_sizes[i] = block_sizes[i];
        }
    }
    /* std::copy(block_sizes.begin(), block_sizes.end(), std::back_inserter(obj->block_sizes)); */

    return obj;
}

uint32_t mram_mem_mgr_t::get_slot(mram_heap_obj_t* obj,
                    uint32_t max_block_size, uint32_t align_size) {
    /*TODO: case for unspecified max. block size */

    uint32_t init_rem_free_block = 64 * MiB;
    uint32_t init_offs = (uint32_t)(-1);
    uint32_t actual_offs = (uint32_t)(-1);
    uint32_t offs_pos = (uint32_t)(-1);

    for (std::size_t pos = 0; pos < (offsets_.size() - 1); pos++) {
        if (slots_[pos].first) { /* free slot */
            uint32_t offs_a = offsets_[pos];
            uint32_t bytes_rem = offs_a % align_size;
            uint32_t offs_b = offs_a + ((align_size - bytes_rem) % align_size);
            uint32_t free_block_size = offsets_[pos + 1] - offs_b;
            uint32_t actual_rem_free_block_size = free_block_size - max_block_size;

            if (actual_rem_free_block_size > 0) { /* TODO: check for equality */
                if (actual_rem_free_block_size < init_rem_free_block) {
                    init_rem_free_block = actual_rem_free_block_size;
                    init_offs = offs_a;
                    actual_offs = offs_b;
                    offs_pos = pos;
                }
            }
        }
    }

    if (actual_offs == ((uint32_t)(-1))) {
        PRINT_ERROR("MRAM block allocation error");
        std::exit(EXIT_FAILURE);
    }
    else {
        offsets_.insert(offsets_.begin() + offs_pos + 1, actual_offs + max_block_size);
        slots_.insert(slots_.begin() + offs_pos + 1, std::make_pair(true, nullptr));

        if (init_offs == actual_offs) {
            assert(slots_[offs_pos].first);
            slots_[offs_pos].first = false;
            slots_[offs_pos].second = obj;
        }
        else {
            offsets_.insert(offsets_.begin() + offs_pos + 1, actual_offs);
            slots_.insert(slots_.begin() + offs_pos + 1, std::make_pair(false, obj));
        }
    }

    return actual_offs;
}

mram_heap_obj_t* mram_mem_mgr_t::get_block(std::string &name) {
    for (auto &obj : heap_->objs) {
        if (obj->name == name) {
            return obj;
        }
    }
    return nullptr;
}

void mram_mem_mgr_t::free_block(std::string &name) {
    uint32_t idx = (uint32_t)(-1);
    mram_heap_obj_t *obj = NULL;

    for (std::size_t i = 0; i < heap_->objs.size(); i++) {
        if (name == heap_->objs.at(i)->name) {
            obj = heap_->objs.at(i);
            idx = i;
            break;
        }
    }

    if (!obj) {
        print_mram_info();
        PRINT_ERROR("%s not found", name.c_str());
        std::exit(EXIT_FAILURE);
    }

    uint32_t offs = obj->offset;
    uint32_t pos = (uint32_t)(-1);

    for (std::size_t p = 0; p < (offsets_.size() - 1); p++) {
        if (offsets_[p] == offs) {
            pos = p;
            break;
        }
    }

    if (pos == (uint32_t)(-1)) {
        PRINT_ERROR("MRAM heap item %u not found", pos);
        std::exit(EXIT_FAILURE);
    }

    if (offs != offsets_[pos]) {
        PRINT_ERROR("MRAM offset error: %u != %u | %s", offs, offsets_[pos], obj->name.c_str());
        std::exit(EXIT_FAILURE);
    }

    if ((offs + obj->max_block_size) != offsets_[pos + 1]) {
        PRINT_ERROR("MRAM offset error: (%u + %u) != %u | %s", offs, obj->max_block_size, offsets_[pos + 1], obj->name.c_str());
        std::exit(EXIT_FAILURE);
    }

    bool prev_free;
    bool post_free;

    if (pos != 0) {
        prev_free = slots_[pos - 1].first;
    }
    else {
        prev_free = false;
    }

    if ((std::size_t)pos != (offsets_.size() - 2)) {
        post_free = slots_[pos + 1].first;
    }
    else { /* the last offset */
        post_free = false;
    }

    if ((prev_free == true) && (post_free == true)) {
        offsets_.erase(offsets_.begin() + pos);
        offsets_.erase(offsets_.begin() + pos + 1);

        slots_.erase(slots_.begin() + pos);
        slots_.erase(slots_.begin() + pos + 1);
    }
    else if ((prev_free == true) && (post_free == false)) {
        offsets_.erase(offsets_.begin() + pos);

        slots_.erase(slots_.begin() + pos);
    }
    else if ((prev_free == false) && (post_free == true)) {
        offsets_.erase(offsets_.begin() + pos + 1);

        slots_.erase(slots_.begin() + pos);
    }
    else if ((prev_free == false) && (post_free == false)) {
        slots_[pos].first = true;
        slots_[pos].second = nullptr;
    }

    delete obj;
    heap_->objs.erase(heap_->objs.begin() + pos);
}

void mram_mem_mgr_t::print_mram_info() {
    PRINT_MSG("Print MRAM info...");
    for (std::size_t pos = 0; pos < (offsets_.size() - 1); pos++) {
        if (slots_[pos].first) {
            PRINT_MSG("Free slot: [%u, %u) %.*f KiB", offsets_[pos], offsets_[pos + 1], 2, (offsets_[pos + 1] - offsets_[pos]) / (float)KiB);
        }
        else {
            mram_heap_obj_t *obj = slots_[pos].second;
            PRINT_MSG("Used slot: [%u, %u) %.*f KiB | %s", offsets_[pos], offsets_[pos + 1], 2, (offsets_[pos + 1] - offsets_[pos]) / (float)KiB, obj->name.c_str());

            if (obj->offset != offsets_[pos]) {
                PRINT_ERROR("MRAM offset error: %u != %u | %s", obj->offset, offsets_[pos], obj->name.c_str());
                std::exit(EXIT_FAILURE);
            }

            if ((obj->offset + obj->max_block_size) != offsets_[pos + 1]) {
                PRINT_ERROR("MRAM offset error: (%u + %u) != %u | %s", obj->offset, obj->max_block_size, offsets_[pos + 1], obj->name.c_str());
                std::exit(EXIT_FAILURE);
            }
        }
    }
}

void timer::start(const std::string &name) {
    if (timing || !cur_event.empty()) {
        PRINT_WARNING("There is an ongoing timing\n");
    }
    else if (events.find(name) != events.end()) {
        auto &e = events[name];
        e.starts.push_back({std::chrono::steady_clock::now()});
        timing = true;
        cur_event = name;
    }
    else {
        event e;
        e.name = name;
        e.starts.push_back({std::chrono::steady_clock::now()});
        events.emplace(std::piecewise_construct, std::forward_as_tuple(name), std::forward_as_tuple(e));
        timing = true;
        cur_event = name;

        events_seq.push_back(name);
    }
}

void timer::stop() {
    if (!timing) {
        PRINT_WARNING("Timer was not started\n");
    }
    else {
        auto &e = events[cur_event];
        e.stops.push_back({std::chrono::steady_clock::now()});
        timing = false;
        cur_event = "";
    }
}

void timer::print() {
    for (auto &n : events_seq) {
        auto &e = events.find(n)->second;
        assert(e.starts.size() == e.stops.size());
        uint64_t total = 0;
        for (std::size_t t = 0; t < e.starts.size(); t++) {
            auto d = std::chrono::duration_cast<std::chrono::microseconds>(e.stops[t] - e.starts[t]).count();
            total += d;
        }
        PRINT_INFO("%s: %f ms", e.name.c_str(), (total / (e.starts.size() * 1000.0)));
    }
}

void timer::print_to_csv(const std::string &f, const std::string &mark, bool append) {
    std::ofstream ofs;
    if (append) {
        ofs.open(f, std::ios_base::app);
    }
    else {
        ofs.open(f);
    }

    if (!ofs) {
        PRINT_WARNING("Cannot open CSV file\n");
    }
    else {
        auto iter = events.begin();
        for (std::size_t t = 0; t < iter->second.starts.size(); t++) {
            /* assuming all events were measured the same number of times */
            for (auto &n : events_seq) {
                auto &e = events.find(n)->second;
                assert(e.starts.size() == e.stops.size());
                auto d = std::chrono::duration_cast<std::chrono::microseconds>(e.stops[t] - e.starts[t]).count();
                ofs << (d / 1000.0) << ",";
            }
            ofs << mark << "\n";
        }
        ofs.flush();
    }

    ofs.close();
}

void transfer_data() {

    try {

        std::random_device rd;
        std::mt19937::result_type seed = rd() ^ (
            (std::mt19937::result_type)
                std::chrono::duration_cast<std::chrono::seconds>(
                    std::chrono::system_clock::now().time_since_epoch()).count() +
            (std::mt19937::result_type)
                std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::system_clock::now().time_since_epoch()).count());

        std::mt19937 uniform_gen(seed);
        std::uniform_int_distribution<uint32_t> uniform_dist(1, 10); /* [x, y] */

        /* allocate DPUs and load binary */
        uint32_t num_dpus, dpuid;
        struct dpu_set_t dpu_set, dpu;
        // uint32_t num_dpus, rankid, dpuid;
        // struct dpu_set_t dpu_set, rank, dpu;

        timer t;
        std::vector<data_xfer_kernel_args_t> dpu_args(NR_DPUS);

        DPU_ASSERT(dpu_alloc(NR_DPUS, NULL, &dpu_set));
        DPU_ASSERT(dpu_get_nr_dpus(dpu_set, &num_dpus));
        assert(num_dpus == NR_DPUS);
        DPU_ASSERT(dpu_load(dpu_set, "./../dpu_bin/dpu_data_xfer", NULL));

        auto num_elems = 100000;
        auto elems_per_dpu = DIVCEIL(num_elems, NR_DPUS);
        elems_per_dpu = (elems_per_dpu % 2 == 0) ? elems_per_dpu : (elems_per_dpu + 1);
        auto elem_size = sizeof(uint32_t);
        auto input_size = elem_size * num_elems;
        auto input_size_per_dpu = elem_size * elems_per_dpu;

        PRINT_TOP_RULE
        PRINT_INFO("DATA TRANSFER");
        PRINT_TOP_RULE
        PRINT_MSG("DPUs: %u", NR_DPUS);
        PRINT_MSG("Tasklets: %u", NR_TASKLETS);
        PRINT_MSG("Input elements: %u", num_elems);
        PRINT_MSG("Elements per DPU: %u", elems_per_dpu);
        PRINT_MSG("Input size: %lu (%f MiB)", input_size, (input_size / (float)(MiB)));
        PRINT_MSG("Input size per DPU: %lu (%f MiB)", input_size_per_dpu, (input_size_per_dpu / (float)(MiB)));

        uint32_t* input_data = (uint32_t*) malloc(input_size);
        for (auto i = 0; i < num_elems; i++) {
            input_data[i] = uniform_dist(uniform_gen);
        }

        uint32_t input_offs = 0;
        std::vector<char*> input_offs_ptrs(NR_DPUS);
        std::vector<uint32_t> input_sizes(NR_DPUS);
        uint8_t *tmp_input_data = (uint8_t*)input_data;

        for (uint32_t d = 0; d < NR_DPUS; d++) {
            if (d == (NR_DPUS - 1)) {
                uint32_t last_size = input_size - input_size_per_dpu * (NR_DPUS - 1);

                input_offs_ptrs[d] = (char*)(tmp_input_data + input_offs);
                input_sizes[d] = last_size;
                input_offs += last_size;
            }
            else {
                input_offs_ptrs[d] = (char*)(tmp_input_data + input_offs);
                input_sizes[d] = input_size_per_dpu;
                input_offs += input_size_per_dpu;
            }
        }

        if (input_offs != input_size) {
            PRINT_ERROR("Error in input data offsets");
            std::exit(EXIT_FAILURE);
        }

        mram_mem_mgr_t* mram_manager = new mram_mem_mgr_t();
        auto *input_block = mram_manager->alloc_block("input data", input_sizes, input_size_per_dpu, 8, 1);

        PRINT_MSG("Transfer data from CPU to DPUs...");
        DPU_FOREACH(dpu_set, dpu, dpuid) {
            DPU_ASSERT(dpu_prepare_xfer(dpu, (void*)(input_offs_ptrs[dpuid])));
        }
        DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, input_block->offset, input_block->max_block_size, DPU_XFER_DEFAULT));

        uint32_t* output_data = (uint32_t*) malloc(input_size);

        uint32_t output_offs = 0;
        std::vector<char*> output_offs_ptrs(NR_DPUS);
        uint8_t *tmp_output_data = (uint8_t*)output_data;

        for (uint32_t d = 0; d < NR_DPUS; d++) {
            if (d == (NR_DPUS - 1)) {
                uint32_t last_size = input_size - input_size_per_dpu * (NR_DPUS - 1);

                output_offs_ptrs[d] = (char*)(tmp_output_data + output_offs);
                output_offs += last_size;
            }
            else {
                output_offs_ptrs[d] = (char*)(tmp_output_data + output_offs);
                output_offs += input_size_per_dpu;
            }
        }

        if (output_offs != input_size) {
            PRINT_ERROR("Error in output data offsets");
            std::exit(EXIT_FAILURE);
        }

        auto *output_block = mram_manager->alloc_block("output data", input_sizes, input_size_per_dpu, 8, 1);
        mram_manager->print_mram_info();

        PRINT_MSG("Transfer parameters to DPUs...");
        DPU_FOREACH(dpu_set, dpu, dpuid) {
            dpu_args[dpuid].output_offset = output_block->offset;
            dpu_args[dpuid].num_elems = output_block->block_sizes[dpuid] / output_block->elem_size;
            DPU_ASSERT(dpu_prepare_xfer(dpu, &dpu_args[dpuid]));
        }
        DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, "dpu_args", 0, sizeof(data_xfer_kernel_args_t), DPU_XFER_DEFAULT));

        PRINT_MSG("Executing the data transfer kernel on DPUs...");
        t.start("DPU exec (data transfer kernel)");
        DPU_ASSERT(dpu_launch(dpu_set, DPU_SYNCHRONOUS)); /* TODO: asynchronous */
        t.stop();

        // PRINT_MSG("Dump DPU logs...");
        // DPU_FOREACH(dpu_set, dpu, dpuid) {
        //     DPU_ASSERT(dpu_log_read(dpu, stdout));
        // }

        PRINT_MSG("Transfer data from DPUs to CPU");
        DPU_FOREACH(dpu_set, dpu, dpuid) {
            DPU_ASSERT(dpu_prepare_xfer(dpu, (void*)(output_offs_ptrs[dpuid])));
        }
        DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_FROM_DPU, DPU_MRAM_HEAP_POINTER_NAME, output_block->offset, output_block->max_block_size, DPU_XFER_DEFAULT));

        for (auto i = 0; i < num_elems; i++) {
            if (input_data[i] != output_data[i]) {
                PRINT_ERROR("Input and output mismatch at idx %u: %u != %u", i, input_data[i], output_data[i]);
                DPU_ASSERT(dpu_free(dpu_set));
                std::exit(EXIT_FAILURE);
            }
        }
        /* PRINT_INFO("Input and Output data equal"); */

        /* free DPUs */
        PRINT_MSG("Free DPUs...");
        DPU_ASSERT(dpu_free(dpu_set));
    }
    catch (const dpu::DpuError &e) {
        std::cerr << e.what() << std::endl;
    }
}
