#include "gif_model.h"
#include "region_filter.h"
#include "split.h"
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Calculate total buffer size needed for a batch of regions
static int calculate_batch_buffer_size(Region *regions, int count) {
  int total_size = 0;
  for (int i = 0; i < count; i++) {
    total_size += 5 * sizeof(int); // metadata
    total_size +=
        regions[i].region_width * regions[i].region_height * sizeof(pixel);
  }
  return total_size;
}

// Pack a batch of regions into a single buffer
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

// Unpack a batch of regions from a single buffer
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

// Slave routine: receives batch of regions from master, processes them, sends
// back
void Slave(void) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // Receive count of regions to process
  int region_count;
  MPI_Recv(&region_count, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  if (region_count <= 0) {
    return;
  }

  Region *regions = (Region *)malloc(region_count * sizeof(Region));
  if (!regions) {
    fprintf(stderr, "Slave %d: Failed to allocate memory for regions\n", rank);
    MPI_Abort(MPI_COMM_WORLD, 1);
    return;
  }

  // Receive buffer size, then the packed buffer containing all regions
  int buffer_size;
  MPI_Recv(&buffer_size, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  char *recv_buffer = (char *)malloc(buffer_size);
  if (!recv_buffer) {
    fprintf(stderr, "Slave %d: Failed to allocate receive buffer\n", rank);
    MPI_Abort(MPI_COMM_WORLD, 1);
    return;
  }

  MPI_Recv(recv_buffer, buffer_size, MPI_PACKED, 0, 2, MPI_COMM_WORLD,
           MPI_STATUS_IGNORE);

  // Unpack all regions from the buffer
  unpack_regions(regions, region_count, recv_buffer, buffer_size,
                 MPI_COMM_WORLD);

  free(recv_buffer);

  // Process each region: apply gray, blur, and sobel filters
  for (int r = 0; r < region_count; r++) {
    apply_all_filters_to_region(&regions[r], 5, 20);
  }

  // Calculate buffer size for sending processed regions back
  int send_buffer_size = calculate_batch_buffer_size(regions, region_count);
  char *send_buffer = (char *)malloc(send_buffer_size);
  if (!send_buffer) {
    fprintf(stderr, "Slave %d: Failed to allocate send buffer\n", rank);
    MPI_Abort(MPI_COMM_WORLD, 1);
    return;
  }

  // Pack all processed regions into single buffer
  pack_regions(regions, region_count, send_buffer, send_buffer_size,
               MPI_COMM_WORLD);

  // Send buffer size, then the packed buffer back to master
  MPI_Send(&send_buffer_size, 1, MPI_INT, 0, 3, MPI_COMM_WORLD);
  MPI_Send(send_buffer, send_buffer_size, MPI_PACKED, 0, 4, MPI_COMM_WORLD);

  free(send_buffer);

  // Free allocated memory for regions
  for (int r = 0; r < region_count; r++) {
    if (regions[r].p) {
      free(regions[r].p);
    }
  }
  free(regions);
}
