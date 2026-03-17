#include "gif_model.h"
#include "region_filter.h"
#include "split.h"
#include "runtime_config.h"
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sobel_cuda.h"

#define TAG_COMMAND 0
#define TAG_REGION_COUNT 1
#define TAG_BUFFER_SIZE 2
#define TAG_BUFFER_DATA 3
#define TAG_RESULT_SIZE 4
#define TAG_RESULT_DATA 5

#define CMD_PROCESS_SPLIT_IMAGE 1
#define CMD_PROCESS_BATCH 2
#define CMD_TERMINATE 3

// Global flag for GPU availability, set once at startup
static int g_use_gpu = 0;

static int calculate_batch_buffer_size(Region *regions, int count) {
  int total_size = 0;
  for (int i = 0; i < count; i++) {
    total_size += 5 * sizeof(int); // metadata
    total_size +=
        regions[i].region_width * regions[i].region_height * sizeof(pixel);
  }
  return total_size;
}

static void pack_regions(Region *regions, int count, char *buffer,
                         int buffer_size, MPI_Comm comm) {
  int position = 0;
  for (int i = 0; i < count; i++) {
    int metadata[5] = {regions[i].image_id, regions[i].region_id,
                       regions[i].region_width, regions[i].region_height,
                       regions[i].k_regions};
    MPI_Pack(metadata, 5, MPI_INT, buffer, buffer_size, &position, comm);

    int pixel_count = regions[i].region_width * regions[i].region_height;
    MPI_Pack(regions[i].p, pixel_count * sizeof(pixel), MPI_BYTE, buffer,
             buffer_size, &position, comm);
  }
}

static void unpack_regions(Region *regions, int count, char *buffer,
                           int buffer_size, MPI_Comm comm) {
  int position = 0;
  for (int i = 0; i < count; i++) {
    int metadata[5];
    MPI_Unpack(buffer, buffer_size, &position, metadata, 5, MPI_INT, comm);

    regions[i].image_id = metadata[0];
    regions[i].region_id = metadata[1];
    regions[i].region_width = metadata[2];
    regions[i].region_height = metadata[3];
    regions[i].k_regions = metadata[4];

    int pixel_count = regions[i].region_width * regions[i].region_height;
    regions[i].p = (pixel *)malloc(pixel_count * sizeof(pixel));

    MPI_Unpack(buffer, buffer_size, &position, regions[i].p,
               pixel_count * sizeof(pixel), MPI_BYTE, comm);
  }
}

// Apply filters with GPU dispatch for all filters if available
static void apply_filters_with_gpu_dispatch(Region *region, int blur_size,
                                            int blur_threshold, runtime_config_t config) {
  apply_all_filters_to_region_gpu(region, blur_size, blur_threshold, g_use_gpu, config);
}

// Apply filters with MPI sync and GPU dispatch for all filters if available
static void apply_filters_mpi_with_gpu_dispatch(Region *region, int blur_size,
                                                int blur_threshold,
                                                MPI_Comm comm, runtime_config_t config) {
  apply_all_filters_to_region_mpi_gpu(region, blur_size, blur_threshold, comm,
                                      g_use_gpu, config);
}

// Handle processing of a split image (with ghost cell synchronization)
static void handle_split_image(int rank, runtime_config_t config) {
  int buffer_size;
  MPI_Recv(&buffer_size, 1, MPI_INT, 0, TAG_BUFFER_SIZE, MPI_COMM_WORLD,
           MPI_STATUS_IGNORE);

  char *recv_buffer = (char *)malloc(buffer_size);
  MPI_Recv(recv_buffer, buffer_size, MPI_PACKED, 0, TAG_BUFFER_DATA,
           MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  Region region;
  unpack_regions(&region, 1, recv_buffer, buffer_size, MPI_COMM_WORLD);
  free(recv_buffer);

  apply_filters_mpi_with_gpu_dispatch(&region, 5, 20, MPI_COMM_WORLD, config);
  int send_buffer_size = calculate_batch_buffer_size(&region, 1);
  char *send_buffer = (char *)malloc(send_buffer_size);
  pack_regions(&region, 1, send_buffer, send_buffer_size, MPI_COMM_WORLD);

  MPI_Send(&send_buffer_size, 1, MPI_INT, 0, TAG_RESULT_SIZE, MPI_COMM_WORLD);
  MPI_Send(send_buffer, send_buffer_size, MPI_PACKED, 0, TAG_RESULT_DATA,
           MPI_COMM_WORLD);

  free(send_buffer);
  free(region.p);
}

static void handle_batch(int rank, runtime_config_t config) {
  int region_count;
  MPI_Recv(&region_count, 1, MPI_INT, 0, TAG_REGION_COUNT, MPI_COMM_WORLD,
           MPI_STATUS_IGNORE);

  if (region_count <= 0) {
    return;
  }

  int buffer_size;
  MPI_Recv(&buffer_size, 1, MPI_INT, 0, TAG_BUFFER_SIZE, MPI_COMM_WORLD,
           MPI_STATUS_IGNORE);

  char *recv_buffer = (char *)malloc(buffer_size);
  MPI_Recv(recv_buffer, buffer_size, MPI_PACKED, 0, TAG_BUFFER_DATA,
           MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  Region *regions = (Region *)malloc(region_count * sizeof(Region));
  unpack_regions(regions, region_count, recv_buffer, buffer_size,
                 MPI_COMM_WORLD);
  free(recv_buffer);

  // #pragma omp parallel for schedule(dynamic) if(openmp_mode != OPENMP_MODE_OFF && region_count > OPENMP_COARSE_THRESHOLD && omp_get_max_threads() > OPENMP_THREADS_THRESHOLD && regions[0].region_width * regions[0].region_height < OPENMP_THRESHOLD)
  for (int r = 0; r < region_count; r++) {
    apply_filters_with_gpu_dispatch(&regions[r], 5, 20, config);
  }

  int send_buffer_size = calculate_batch_buffer_size(regions, region_count);
  char *send_buffer = (char *)malloc(send_buffer_size);
  pack_regions(regions, region_count, send_buffer, send_buffer_size,
               MPI_COMM_WORLD);

  MPI_Send(&send_buffer_size, 1, MPI_INT, 0, TAG_RESULT_SIZE, MPI_COMM_WORLD);
  MPI_Send(send_buffer, send_buffer_size, MPI_PACKED, 0, TAG_RESULT_DATA,
           MPI_COMM_WORLD);

  free(send_buffer);

  for (int r = 0; r < region_count; r++) {
    if (regions[r].p) {
      free(regions[r].p);
    }
  }
  free(regions);
}

void Slave(runtime_config_t config) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  g_use_gpu = has_nvidia_gpu();
  if (g_use_gpu) {
#ifdef USE_CUDA
    char device_name[256];
    cuda_get_device_name(device_name, sizeof(device_name));
    printf("[Rank %d] CUDA GPU detected: %s\n",
           rank, device_name);
#endif
  } else {
    printf("[Rank %d] No CUDA GPU - using OpenMP for all filters\n", rank);
  }

  while (1) {
    int cmd;
    MPI_Recv(&cmd, 1, MPI_INT, 0, TAG_COMMAND, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);

    switch (cmd) {
    case CMD_PROCESS_SPLIT_IMAGE:
      handle_split_image(rank, config);
      break;

    case CMD_PROCESS_BATCH:
      handle_batch(rank, config);
      break;

    case CMD_TERMINATE:
      return;

    default:
      fprintf(stderr, "Slave %d: Unknown command %d\n", rank, cmd);
      MPI_Abort(MPI_COMM_WORLD, 1);
      return;
    }
  }
}
