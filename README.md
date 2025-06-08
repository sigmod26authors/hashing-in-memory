# Hashing-in-Memory: Leveraging Processing-in-Memory for Hash Indexes

This repository contains the source code for the paper, "Hashing-in-Memory: Leveraging Processing-in-Memory for Hash Indexes", submitted to the Round 2 paper submissions of SIGMOD 2026.


## Installing the UPMEM DPU toolchain

Please refer to the [UPMEM SDK](https://sdk.upmem.com/) for the installation of the UPMEM DPU toolchain.


## Building the DPU program
```bash
make all
```

## Building the host program
```bash
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make
```

## Running the PIM application
```bash
./pimindex
```

## Directories

* [`dpu/`](./dpu) contains source files for the DPU program
* [`host/`](./host) contains source files for the host program
* [`include/`](./include) contains header files for the DPU and host programs
* [`baselines/`](./baselines) contains source files for CPU baselines


## Configurations

| Configurations | Files |
| --- | --- |
| Design alternatives | [`CMakeLists.txt`](./CMakeLists.txt) [`Makefile`](./Makefile) [`main.cpp`](./main.cpp) [`include/pimindex.h`](./include/pimindex.h) [`run.sh`](./run.sh) |
| Implementation parameters | [`include/<implementation>.h`](./include/) |
| DPU parameters | [`Makefile`](./Makefile) [`include/<implementation>.h`](./include/) [`run.sh`](./run.sh) |
