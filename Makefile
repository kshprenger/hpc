SRC_DIR=src
PARALLEL_SRC_DIR=src/parallel
HEADER_DIR=include
OBJ_DIR=obj
OBJ_DIR_PARALLEL=obj_parallel
OBJ_DIR_NONCUDA=obj_noncuda

CC=gcc
MPICC=mpicc
NVCC=nvcc
CFLAGS=-O3 -I$(HEADER_DIR)
CFLAGS_PARALLEL=-O3 -I$(HEADER_DIR) -fopenmp
LDFLAGS=-lm
LDFLAGS_PARALLEL=-lm -fopenmp

CUDA_PATH ?= /usr/local/cuda
MPI_INCDIR := $(shell mpicc --showme:incdirs 2>/dev/null | tr ' ' '\n' | head -1)
NVCCFLAGS=-O3 -I$(HEADER_DIR) $(if $(MPI_INCDIR),-I$(MPI_INCDIR)) \
          -Xcompiler -fPIC \
          -gencode arch=compute_89,code=sm_89 \
          -gencode arch=compute_86,code=sm_86 \
          -gencode arch=compute_75,code=sm_75
CUDA_LDFLAGS=-L$(CUDA_PATH)/lib64 -lcudart -lstdc++

CUDA_AVAILABLE := $(shell which nvcc > /dev/null 2>&1 && echo 1 || echo 0)

ifeq ($(CUDA_AVAILABLE),1)
    CFLAGS_PARALLEL += -DUSE_CUDA
    LDFLAGS_PARALLEL += $(CUDA_LDFLAGS)
    CUDA_OBJ = $(OBJ_DIR_PARALLEL)/sobel_filter_cuda.o
else
    CUDA_OBJ =
endif

# Common source files (GIF library)
COMMON_SRC= dgif_lib.c \
	egif_lib.c \
	gif_err.c \
	gif_font.c \
	gif_hash.c \
	gifalloc.c \
	openbsd-reallocarray.c \
	quantize.c

# Serial version object files
OBJ= $(OBJ_DIR)/dgif_lib.o \
	$(OBJ_DIR)/egif_lib.o \
	$(OBJ_DIR)/gif_err.o \
	$(OBJ_DIR)/gif_font.o \
	$(OBJ_DIR)/gif_hash.o \
	$(OBJ_DIR)/gifalloc.o \
	$(OBJ_DIR)/main.o \
	$(OBJ_DIR)/openbsd-reallocarray.o \
	$(OBJ_DIR)/quantize.o

# Parallel version object files
OBJ_PARALLEL= $(OBJ_DIR_PARALLEL)/dgif_lib.o \
	$(OBJ_DIR_PARALLEL)/egif_lib.o \
	$(OBJ_DIR_PARALLEL)/gif_err.o \
	$(OBJ_DIR_PARALLEL)/gif_font.o \
	$(OBJ_DIR_PARALLEL)/gif_hash.o \
	$(OBJ_DIR_PARALLEL)/gifalloc.o \
	$(OBJ_DIR_PARALLEL)/parallel_main.o \
	$(OBJ_DIR_PARALLEL)/mpi_master.o \
	$(OBJ_DIR_PARALLEL)/mpi_slave.o \
	$(OBJ_DIR_PARALLEL)/blur_filter.o \
	$(OBJ_DIR_PARALLEL)/grey_filter.o \
	$(OBJ_DIR_PARALLEL)/image_load.o \
	$(OBJ_DIR_PARALLEL)/image_store.o \
	$(OBJ_DIR_PARALLEL)/sobel_filter.o \
	$(OBJ_DIR_PARALLEL)/openbsd-reallocarray.o \
	$(OBJ_DIR_PARALLEL)/quantize.o \
	$(CUDA_OBJ)

all: $(OBJ_DIR) $(OBJ_DIR_PARALLEL) sobelf parallel_sobelf

noncuda: $(OBJ_DIR_NONCUDA) parallel_sobelf_noncuda

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

$(OBJ_DIR_PARALLEL):
	mkdir -p $(OBJ_DIR_PARALLEL)

$(OBJ_DIR_NONCUDA):
	mkdir -p $(OBJ_DIR_NONCUDA)

$(OBJ_DIR)/%.o : $(SRC_DIR)/%.c
	$(CC) $(CFLAGS) -c -o $@ $<

$(OBJ_DIR_PARALLEL)/%.o : $(SRC_DIR)/%.c
	$(MPICC) $(CFLAGS_PARALLEL) -c -o $@ $<

$(OBJ_DIR_PARALLEL)/%.o : $(PARALLEL_SRC_DIR)/%.c
	$(MPICC) $(CFLAGS_PARALLEL) -c -o $@ $<

# Non-CUDA build rules: compile with plain OpenMP flags (no -DUSE_CUDA)
CFLAGS_NONCUDA=-O3 -I$(HEADER_DIR) -fopenmp
LDFLAGS_NONCUDA=-lm -fopenmp

$(OBJ_DIR_NONCUDA)/%.o : $(SRC_DIR)/%.c
	$(MPICC) $(CFLAGS_NONCUDA) -c -o $@ $<

$(OBJ_DIR_NONCUDA)/%.o : $(PARALLEL_SRC_DIR)/%.c
	$(MPICC) $(CFLAGS_NONCUDA) -c -o $@ $<

$(OBJ_DIR_PARALLEL)/sobel_filter_cuda.o : $(PARALLEL_SRC_DIR)/sobel_filter.cu
	$(NVCC) $(NVCCFLAGS) -c -o $@ $<

sobelf: $(OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

parallel_sobelf: $(OBJ_PARALLEL)
	$(MPICC) $(CFLAGS_PARALLEL) -o $@ $^ $(LDFLAGS_PARALLEL)

OBJ_NONCUDA= $(OBJ_DIR_NONCUDA)/dgif_lib.o \
	$(OBJ_DIR_NONCUDA)/egif_lib.o \
	$(OBJ_DIR_NONCUDA)/gif_err.o \
	$(OBJ_DIR_NONCUDA)/gif_font.o \
	$(OBJ_DIR_NONCUDA)/gif_hash.o \
	$(OBJ_DIR_NONCUDA)/gifalloc.o \
	$(OBJ_DIR_NONCUDA)/parallel_main.o \
	$(OBJ_DIR_NONCUDA)/mpi_master.o \
	$(OBJ_DIR_NONCUDA)/mpi_slave.o \
	$(OBJ_DIR_NONCUDA)/blur_filter.o \
	$(OBJ_DIR_NONCUDA)/grey_filter.o \
	$(OBJ_DIR_NONCUDA)/image_load.o \
	$(OBJ_DIR_NONCUDA)/image_store.o \
	$(OBJ_DIR_NONCUDA)/sobel_filter.o \
	$(OBJ_DIR_NONCUDA)/openbsd-reallocarray.o \
	$(OBJ_DIR_NONCUDA)/quantize.o

parallel_sobelf_noncuda: $(OBJ_NONCUDA)
	$(MPICC) $(CFLAGS_NONCUDA) -o $@ $^ $(LDFLAGS_NONCUDA)

cuda: clean_parallel $(OBJ_DIR_PARALLEL) parallel_sobelf

clean_parallel:
	rm -f parallel_sobelf
	rm -rf $(OBJ_DIR_PARALLEL)

clean_noncuda:
	rm -f parallel_sobelf_noncuda
	rm -rf $(OBJ_DIR_NONCUDA)

clean:
	rm -f sobelf parallel_sobelf parallel_sobelf_noncuda
	rm -rf $(OBJ_DIR) $(OBJ_DIR_PARALLEL) $(OBJ_DIR_NONCUDA)
	./clean_test.sh

.PHONY: all noncuda clean clean_parallel clean_noncuda cuda
