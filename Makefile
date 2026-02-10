SRC_DIR=src
PARALLEL_SRC_DIR=src/parallel
HEADER_DIR=include
OBJ_DIR=obj
OBJ_DIR_PARALLEL=obj_parallel

CC=gcc
MPICC=mpicc
CFLAGS=-O3 -I$(HEADER_DIR)
CFLAGS_PARALLEL=-O3 -I$(HEADER_DIR)
LDFLAGS=-lm
LDFLAGS_PARALLEL=-lm

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
	$(OBJ_DIR_PARALLEL)/blur_filter.o \
	$(OBJ_DIR_PARALLEL)/grey_filter.o \
	$(OBJ_DIR_PARALLEL)/image_load.o \
	$(OBJ_DIR_PARALLEL)/image_store.o \
	$(OBJ_DIR_PARALLEL)/sobel_filter.o \
	$(OBJ_DIR_PARALLEL)/openbsd-reallocarray.o \
	$(OBJ_DIR_PARALLEL)/quantize.o

all: $(OBJ_DIR) $(OBJ_DIR_PARALLEL) sobelf parallel_sobelf

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

$(OBJ_DIR_PARALLEL):
	mkdir -p $(OBJ_DIR_PARALLEL)

# Serial version compilation rules
$(OBJ_DIR)/%.o : $(SRC_DIR)/%.c
	$(CC) $(CFLAGS) -c -o $@ $<

# Parallel version compilation rules (using mpicc)
$(OBJ_DIR_PARALLEL)/%.o : $(SRC_DIR)/%.c
	$(MPICC) $(CFLAGS_PARALLEL) -c -o $@ $<

$(OBJ_DIR_PARALLEL)/%.o : $(PARALLEL_SRC_DIR)/%.c
	$(MPICC) $(CFLAGS_PARALLEL) -c -o $@ $<

# Link serial version
sobelf: $(OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# Link parallel version (using mpicc)
parallel_sobelf: $(OBJ_PARALLEL)
	$(MPICC) $(CFLAGS_PARALLEL) -o $@ $^ $(LDFLAGS_PARALLEL)

clean:
	rm -f sobelf parallel_sobelf
	rm -rf $(OBJ_DIR) $(OBJ_DIR_PARALLEL)
	./clean_test.sh
