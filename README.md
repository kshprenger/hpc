# README

## Build instructions

This project can build three executables:

- `sobelf` — serial version
- `parallel_sobelf` — parallel version with MPI/OpenMP, and CUDA if available
- `parallel_sobelf_noncuda` — parallel version with MPI/OpenMP only

## Requirements

You need:

- `make`
- `gcc`
- `mpicc`

For CUDA support, you also need:

- `nvcc`
- CUDA Toolkit

By default, CUDA is expected at:

`/usr/local/cuda`

If needed, you can override this path with:

    make CUDA_PATH=/path/to/cuda

## How to build

### Build the serial version and the default parallel version

    make

This builds:

- `sobelf`
- `parallel_sobelf`

When building `parallel_sobelf`, the Makefile automatically checks whether `nvcc` is installed:

- if CUDA is available, CUDA support is included
- otherwise, it builds the parallel version without CUDA

### Build the parallel version without CUDA

    make noncuda

This builds:

- `parallel_sobelf_noncuda`

Use this target when you want the MPI/OpenMP version only, without any CUDA dependency.

### Rebuild the default parallel version

    make cuda

This cleans the previous default parallel build and rebuilds:

- `parallel_sobelf`

If CUDA is available, this build includes CUDA support.

## What each executable uses

### `sobelf`

Built with `gcc`.

### `parallel_sobelf`

Built with `mpicc` and OpenMP support enabled.

If `nvcc` is detected, CUDA code is also compiled and linked into this executable.

### `parallel_sobelf_noncuda`

Built with `mpicc` and OpenMP support enabled, but without CUDA.

## Object file directories

The Makefile stores object files in separate directories:

- `obj/` for the serial build
- `obj_parallel/` for the default parallel build
- `obj_noncuda/` for the non-CUDA parallel build

## Cleaning

### Clean everything

    make clean
    
It also runs:

    ./clean_test.sh

### Clean only the default parallel build

    make clean_parallel

### Clean only the non-CUDA parallel build

    make clean_noncuda

## Summary

Build everything:

    make

Build non-CUDA parallel version:

    make noncuda

Rebuild default parallel version:

    make cuda

Clean everything:

    make clean
## Running the program

The parallel executable is launched with `mpirun` and takes an input GIF and an output GIF as arguments.

Usage:

    mpirun -np <num_processes> ./parallel_sobelf [--mpi off|auto|full|hybrid] [--openmp off|auto|force] [--cuda off|auto|force] input.gif output.gif

Example:

    mpirun -np 4 ./parallel_sobelf --mpi auto --openmp auto --cuda auto input.gif output.gif

If no flags are provided, the default mode for **MPI**, **OpenMP**, and **CUDA** is `auto`.

- `off` disables the corresponding method entirely.
- `auto` lets the program decide automatically whether using that method is worthwhile.
- `force` forces the use of that specific method when that mode exists.

More specifically:

- `--mpi off` disables MPI-based parallelization.
- `--mpi full` forces MPI frame batching only, without image splitting.
- `--mpi hybrid` forces the hybrid MPI strategy: frames are distributed in batches, and if the number of frames is not divisible by the number of MPI ranks, the remainder is processed using image splitting.
- `--openmp off` disables OpenMP.
- `--openmp force` forces the use of OpenMP.
- `--cuda off` disables CUDA.
- `--cuda force` forces the use of CUDA.

For hybrid **MPI + OpenMP** execution, we specifically recommend launching the program with:

    OMP_PROC_BIND=false mpirun --bind-to none -np <num_processes> ./parallel_sobelf ...

This avoids overly strict process/thread binding and gave better performance in our tests when MPI and OpenMP were used together.

The serial version can be run as:

    ./sobelf input.gif output.gif
