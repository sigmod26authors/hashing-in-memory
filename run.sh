#!/bin/bash

dpu_count=(64)
tasklet_count=(16)
operation_count=(1) # (million)

MAKEFILE=$PWD/Makefile
# INCLUDE_FILE=$PWD/include/pimindex_extendible.h
# INCLUDE_FILE=$PWD/include/pimindex_cuckoo.h
# INCLUDE_FILE=$PWD/include/pimindex_direct_mapping.h
# INCLUDE_FILE=$PWD/include/three_level.h
INCLUDE_FILE=$PWD/include/cooperative.h

for operations in "${operation_count[@]}"; do
    # set number of index operations
    sed -i 's/\(.*define NR_OPERATIONS \)\([0-9]\+\)\(.*\)/\1'"$operations"'\3/' $INCLUDE_FILE

    for dpus in "${dpu_count[@]}"; do
        # set number of DPUs
        sed -i 's/\(.*define NR_DPUS \)\([0-9]\+\)\(.*\)/\1'"$dpus"'\3/' $INCLUDE_FILE

        for tasklets in "${tasklet_count[@]}"; do
            # set number of tasklets
            sed -i 's/\(.*define NR_TASKLETS \)\([0-9]\+\)\(.*\)/\1'"$tasklets"'\3/' $INCLUDE_FILE

            # build DPU program
            make clean -f $MAKEFILE
            make all -f $MAKEFILE NR_DPUS="$dpus" NR_TASKLETS="$tasklets"

            # build host program
            if [ ! -d build ]; then
                mkdir build
                cd build
                cmake -DCMAKE_BUILD_TYPE=Release ..
            else
                cd build
            fi
            make pimindex

            # run PIM application
            ./pimindex
        done
    done
done
