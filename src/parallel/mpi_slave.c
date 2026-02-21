#include "gif_model.h"
#include "region_filter.h"
#include "split.h"
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Slave routine: receives regions from master, processes them, sends back
void Slave(void) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // Receive count of regions to process
  int region_count;
  MPI_Recv(&region_count, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  if (region_count <= 0) {
    return;
  }

  // Allocate array to hold regions
  Region *regions = (Region *)malloc(region_count * sizeof(Region));
  if (!regions) {
    fprintf(stderr, "Slave %d: Failed to allocate memory for regions\n", rank);
    MPI_Abort(MPI_COMM_WORLD, 1);
    return;
  }

  // Receive each region from master
  for (int r = 0; r < region_count; r++) {
    // Receive region metadata
    int metadata[5];
    MPI_Recv(metadata, 5, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    regions[r].image_id = metadata[0];
    regions[r].region_id = metadata[1];
    regions[r].region_width = metadata[2];
    regions[r].region_height = metadata[3];
    regions[r].k_regions = metadata[4];

    // Allocate and receive pixel data
    int pixel_count = regions[r].region_width * regions[r].region_height;
    regions[r].p = (pixel *)malloc(pixel_count * sizeof(pixel));
    if (!regions[r].p) {
      fprintf(stderr, "Slave %d: Failed to allocate memory for pixels\n", rank);
      MPI_Abort(MPI_COMM_WORLD, 1);
      return;
    }

    MPI_Recv(regions[r].p, pixel_count * sizeof(pixel), MPI_BYTE, 0, 2,
             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }

  // Process each region: apply gray, blur, and sobel filters
  for (int r = 0; r < region_count; r++) {
    apply_all_filters_to_region(&regions[r], 5, 20);
  }

  // Send processed regions back to master
  for (int r = 0; r < region_count; r++) {
    // Send region metadata
    int metadata[5] = {regions[r].image_id, regions[r].region_id,
                       regions[r].region_width, regions[r].region_height,
                       regions[r].k_regions};
    MPI_Send(metadata, 5, MPI_INT, 0, 3, MPI_COMM_WORLD);

    // Send pixel data
    int pixel_count = regions[r].region_width * regions[r].region_height;
    MPI_Send(regions[r].p, pixel_count * sizeof(pixel), MPI_BYTE, 0, 4,
             MPI_COMM_WORLD);
  }

  // Free allocated memory
  for (int r = 0; r < region_count; r++) {
    if (regions[r].p) {
      free(regions[r].p);
    }
  }
  free(regions);
}
