#include "pimindex.h"
#include "pimindex.hpp"


int main(int argc, char *argv[]) {
    void *workload;
    uint32_t num_keys = NR_OPERATIONS * 2; /* positive and negative search */
#ifdef DATA_FILE
    std::string insert_file(DATA_FILE);
    insert_file += "_insert";
    std::string search_file(DATA_FILE);
    search_file += "_search";
    // search_file += "_nsearch";
    auto ifile = access(insert_file.c_str(), F_OK) == 0;
    auto sfile = access(search_file.c_str(), F_OK) == 0;
    if ((ifile && !sfile) || (!ifile && sfile)) {
        std::cerr << "incorrect insert/search file...\n";
        return 1;
    }
    else if (!ifile && !sfile) {
        workload = generate_workload(num_keys);
    }
    else {
        workload = malloc(num_keys * sizeof(uint32_t));
    }
#else
    workload = generate_workload(num_keys);
#endif
    run_cooperative(workload, NR_LOADS, NR_OPERATIONS);
    // run_three_level(workload, NR_LOADS, NR_OPERATIONS);
    // run_pimindex_extendible(workload, NR_LOADS, NR_OPERATIONS);
    // run_pimindex_extendible_batch(workload, NR_LOADS, NR_OPERATIONS);
    // run_pimindex_direct_mapping(workload, NR_LOADS, NR_OPERATIONS);
    // run_pimindex_cuckoo(workload, NR_LOADS, NR_OPERATIONS);
    // run_pimindex_cuckoo_batch(workload, NR_LOADS, NR_OPERATIONS);
    free(workload);
    return 0;
}
