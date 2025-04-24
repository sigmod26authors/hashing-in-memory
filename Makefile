SRC_DIR ?= .
DPU_DIR := ${SRC_DIR}/dpu
HOST_DIR := ${SRC_DIR}/host
INCL_DIR := ${SRC_DIR}/include

BIN_DIR := ${SRC_DIR}/dpu_bin
CONF_DIR := ${SRC_DIR}/dpu_conf

NR_DPUS ?= 64
NR_TASKLETS ?= 16

__dirs := $(shell mkdir -p ${BIN_DIR} ${CONF_DIR})

define conf_filename
	${CONF_DIR}/.NR_DPUS_$(1)_NR_TASKLETS_$(2).conf
endef
CONF := $(call conf_filename,${NR_DPUS},${NR_TASKLETS})

PIMINDEX := ${BIN_DIR}/dpu_pimindex
PIMINDEX2 := ${BIN_DIR}/dpu_pimindex2

# PIMINDEX_SRC := ${DPU_DIR}/pimindex_extendible.c
# PIMINDEX_SRC := ${DPU_DIR}/pimindex_cuckoo.c
# PIMINDEX_SRC := ${DPU_DIR}/pimindex_direct_mapping.c
# PIMINDEX_SRC := ${DPU_DIR}/three_level1.c
# PIMINDEX_SRC2 := ${DPU_DIR}/three_level2.c
PIMINDEX_SRC := ${DPU_DIR}/cooperative1.c
PIMINDEX_SRC2 := ${DPU_DIR}/cooperative2.c

.PHONY: all clean test

FLAGS := -Wall -Wextra -Werror -g -I${INCL_DIR}
DPU_FLAGS := ${FLAGS} -O2 -DNR_TASKLETS=${NR_TASKLETS}

# all: ${PIMINDEX}
all: ${PIMINDEX} ${PIMINDEX2}

${CONF}:
	$(RM) $(call conf_filename,*,*)
	touch $(CONF)

${PIMINDEX}: ${PIMINDEX_SRC} ${INCL_DIR} ${CONF}
	dpu-upmem-dpurte-clang ${DPU_FLAGS} -o $@ ${PIMINDEX_SRC}

${PIMINDEX2}: ${PIMINDEX_SRC2} ${INCL_DIR} ${CONF}
	dpu-upmem-dpurte-clang ${DPU_FLAGS} -o $@ ${PIMINDEX_SRC2}

clean:
#	$(RM) ${PIMINDEX}
	$(RM) ${PIMINDEX} ${PIMINDEX2}

test: ${PIMINDEX}
	dpu-lldb -f ${PIMINDEX}
