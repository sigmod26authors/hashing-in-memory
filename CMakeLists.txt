project(pimindex)
cmake_minimum_required(VERSION 3.14)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-long-long -fPIC -march=native")
set(CMAKE_ENABLE_COMPILE_COMMANDS "ON")

if (${CMAKE_BUILD_TYPE} STREQUAL "Release")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -DNDEBUG -Ofast")
endif ()

add_library(dpu SHARED IMPORTED)
set(DPU_LIB "-ldpu")
set(PIMINDEX_SOURCES
  pimindex.cpp
  host/cooperative.cpp
  # host/three_level.cpp
  # host/pimindex_extendible.cpp
  # host/pimindex_extendible_batch.cpp
  # host/pimindex_cuckoo.cpp
  # host/pimindex_cuckoo_batch.cpp
  # host/pimindex_direct_mapping.cpp
)
set(PIMINDEX_INCLUDES include /usr/include/dpu)
set(PIMINDEX_LIBRARIES ${DPU_LIB} pthread)

add_executable(pimindex main.cpp ${PIMINDEX_SOURCES})
target_include_directories(pimindex PRIVATE ${PIMINDEX_INCLUDES} ${PROJECT_SOURCE_DIR})
target_link_libraries(pimindex PRIVATE ${PIMINDEX_LIBRARIES})
