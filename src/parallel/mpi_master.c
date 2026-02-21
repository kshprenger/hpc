#include "gif_model.h"
#include "persist_api.h"
#include "region_filter.h"
#include "split.h"
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

// Forward declaration of Slave function
extern void Slave(void);

// Helper function to compare regions for sorting by image_id then region_id
static int compare_regions(const void *a, const void *b) {
  const Region *ra = (const Region *)a;
  const Region *rb = (const Region *)b;
  if (ra->image_id != rb->image_id) {
    return ra->image_id - rb->image_id;
  }
  return ra->region_id - rb->region_id;
}

// Calculate total buffer size needed for a batch of regions
static int calculate_batch_buffer_size(Region *regions, int count) {
  // For each region: 5 ints (metadata) + pixel data
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

void Master(char *input_file, char *output_file) {
  int rank, world_size;
  animated_gif *image = NULL;
  struct timeval t1, t2;
  double duration;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  // ==================== LOAD GIF ====================
  gettimeofday(&t1, NULL);

  image = load_pixels(input_file);
  if (image == NULL) {
    fprintf(stderr, "Master: Failed to load GIF from %s\n", input_file);
    MPI_Abort(MPI_COMM_WORLD, 1);
    return;
  }

  gettimeofday(&t2, NULL);
  duration = (t2.tv_sec - t1.tv_sec) + ((t2.tv_usec - t1.tv_usec) / 1e6);
  printf("GIF loaded from file %s with %d image(s) in %lf s\n", input_file,
         image->n_images, duration);

  int n_images = image->n_images;

  // ==================== SPECIAL CASE: Single rank ====================
  if (world_size == 1) {
    gettimeofday(&t1, NULL);

    // Process all images locally without any MPI communication
    for (int i = 0; i < n_images; i++) {
      // Create a single region for the entire image (k_regions = 1)
      Region *regions =
          Split(image->p[i], i, image->width[i], image->height[i], 1);

      // Apply filters to the region
      apply_all_filters_to_region(&regions[0], 5, 20);

      // Copy processed pixels back to image
      pixel *combined = Combine(regions, image->width[i], image->height[i], 1);
      free(image->p[i]);
      image->p[i] = combined;

      // Free region memory
      free(regions[0].p);
      free(regions);
    }

    gettimeofday(&t2, NULL);
    duration = (t2.tv_sec - t1.tv_sec) + ((t2.tv_usec - t1.tv_usec) / 1e6);
    printf("SOBEL done in %lf s\n", duration);

    // Store output
    gettimeofday(&t1, NULL);

    if (!store_pixels(output_file, image)) {
      fprintf(stderr, "Master: Failed to store GIF to %s\n", output_file);
      return;
    }

    gettimeofday(&t2, NULL);
    duration = (t2.tv_sec - t1.tv_sec) + ((t2.tv_usec - t1.tv_usec) / 1e6);
    printf("Export done in %lf s in file %s\n", duration, output_file);

    return;
  }

  // ==================== MULTI-RANK: Create and distribute regions
  // ====================
  gettimeofday(&t1, NULL);

  // Prepare arrays to collect regions for each worker
  Region **worker_regions = (Region **)malloc(world_size * sizeof(Region *));
  int *worker_region_count = (int *)calloc(world_size, sizeof(int));
  int *worker_region_capacity = (int *)malloc(world_size * sizeof(int));

  for (int w = 0; w < world_size; w++) {
    worker_region_capacity[w] = 16;
    worker_regions[w] =
        (Region *)malloc(worker_region_capacity[w] * sizeof(Region));
  }

  // Track total regions created for later receiving
  int total_regions = 0;

  // Round-robin counter for images that aren't split
  int rr_counter = 0;

  // Iterate over all images and distribute regions
  for (int i = 0; i < n_images; i++) {
    Region *regions;
    int k_regions;

    if (n_images < world_size) {
      // If images < ranks, split each image by rank count
      k_regions = world_size;
      regions =
          Split(image->p[i], i, image->width[i], image->height[i], k_regions);
    } else {
      // If images >= ranks, only split remainder images
      int remainder = n_images % world_size;
      if (i < remainder) {
        k_regions = world_size;
        regions =
            Split(image->p[i], i, image->width[i], image->height[i], k_regions);
      } else {
        // No splitting - treat entire image as one region
        k_regions = 1;
        regions =
            Split(image->p[i], i, image->width[i], image->height[i], k_regions);
      }
    }

    total_regions += k_regions;

    // Distribute regions to workers
    if (k_regions == world_size) {
      // Each region goes to corresponding worker rank
      for (int r = 0; r < k_regions; r++) {
        int worker_id = r;

        // Expand capacity if needed
        if (worker_region_count[worker_id] >=
            worker_region_capacity[worker_id]) {
          worker_region_capacity[worker_id] *= 2;
          worker_regions[worker_id] = (Region *)realloc(
              worker_regions[worker_id],
              worker_region_capacity[worker_id] * sizeof(Region));
        }

        worker_regions[worker_id][worker_region_count[worker_id]] = regions[r];
        worker_region_count[worker_id]++;
      }
    } else {
      // k_regions == 1, assign whole image to a worker (round-robin)
      int worker_id = rr_counter % world_size;
      rr_counter++;

      // Expand capacity if needed
      if (worker_region_count[worker_id] >= worker_region_capacity[worker_id]) {
        worker_region_capacity[worker_id] *= 2;
        worker_regions[worker_id] = (Region *)realloc(
            worker_regions[worker_id],
            worker_region_capacity[worker_id] * sizeof(Region));
      }

      worker_regions[worker_id][worker_region_count[worker_id]] = regions[0];
      worker_region_count[worker_id]++;
    }

    // Free the regions array (but not the pixel data - it's now owned by
    // worker_regions)
    free(regions);
  }

  // ==================== SEND BATCH OF REGIONS TO WORKERS (rank > 0)
  // ====================
  for (int w = 1; w < world_size; w++) {
    int count = worker_region_count[w];

    // Send count of regions first
    MPI_Send(&count, 1, MPI_INT, w, 0, MPI_COMM_WORLD);

    if (count > 0) {
      // Calculate buffer size and pack all regions into one buffer
      int buffer_size = calculate_batch_buffer_size(worker_regions[w], count);
      char *send_buffer = (char *)malloc(buffer_size);

      pack_regions(worker_regions[w], count, send_buffer, buffer_size,
                   MPI_COMM_WORLD);

      // Send buffer size, then the packed buffer
      MPI_Send(&buffer_size, 1, MPI_INT, w, 1, MPI_COMM_WORLD);
      MPI_Send(send_buffer, buffer_size, MPI_PACKED, w, 2, MPI_COMM_WORLD);

      free(send_buffer);

      // Free the pixel data after sending
      for (int r = 0; r < count; r++) {
        free(worker_regions[w][r].p);
        worker_regions[w][r].p = NULL;
      }
    }
  }

  // ==================== PROCESS MASTER'S OWN REGIONS LOCALLY
  // ====================
  int master_count = worker_region_count[0];
  Region *master_regions = worker_regions[0];

  for (int r = 0; r < master_count; r++) {
    apply_all_filters_to_region(&master_regions[r], 5, 20);
  }

  // ==================== RECEIVE PROCESSED REGIONS FROM WORKERS
  // ====================
  Region *all_processed_regions =
      (Region *)malloc(total_regions * sizeof(Region));
  int processed_idx = 0;

  // Copy master's processed regions
  for (int r = 0; r < master_count; r++) {
    all_processed_regions[processed_idx++] = master_regions[r];
  }

  // Receive processed regions from each worker (single recv per worker)
  for (int w = 1; w < world_size; w++) {
    int count = worker_region_count[w];

    if (count > 0) {
      // Receive buffer size, then the packed buffer
      int buffer_size;
      MPI_Recv(&buffer_size, 1, MPI_INT, w, 3, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);

      char *recv_buffer = (char *)malloc(buffer_size);
      MPI_Recv(recv_buffer, buffer_size, MPI_PACKED, w, 4, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);

      // Unpack regions from buffer
      unpack_regions(&all_processed_regions[processed_idx], count, recv_buffer,
                     buffer_size, MPI_COMM_WORLD);

      free(recv_buffer);
      processed_idx += count;
    }
  }

  gettimeofday(&t2, NULL);
  duration = (t2.tv_sec - t1.tv_sec) + ((t2.tv_usec - t1.tv_usec) / 1e6);
  printf("SOBEL done in %lf s\n", duration);

  // ==================== COMBINE REGIONS BACK INTO IMAGES ====================
  gettimeofday(&t1, NULL);

  // Sort all regions by image_id then region_id
  qsort(all_processed_regions, total_regions, sizeof(Region), compare_regions);

  // Group regions by image_id and combine
  int region_idx = 0;
  for (int i = 0; i < n_images; i++) {
    // Find all regions for this image
    int image_region_count = 0;
    int start_idx = region_idx;

    while (region_idx < total_regions &&
           all_processed_regions[region_idx].image_id == i) {
      image_region_count++;
      region_idx++;
    }

    if (image_region_count == 0) {
      fprintf(stderr, "Master: No regions found for image %d\n", i);
      continue;
    }

    // Combine regions back into the image
    int k_regions = all_processed_regions[start_idx].k_regions;
    pixel *combined = Combine(&all_processed_regions[start_idx],
                              image->width[i], image->height[i], k_regions);

    // Replace the original image pixels
    free(image->p[i]);
    image->p[i] = combined;

    // Free region pixel data
    for (int r = start_idx; r < start_idx + image_region_count; r++) {
      if (all_processed_regions[r].p) {
        free(all_processed_regions[r].p);
        all_processed_regions[r].p = NULL;
      }
    }
  }

  // ==================== STORE OUTPUT GIF ====================
  if (!store_pixels(output_file, image)) {
    fprintf(stderr, "Master: Failed to store GIF to %s\n", output_file);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  gettimeofday(&t2, NULL);
  duration = (t2.tv_sec - t1.tv_sec) + ((t2.tv_usec - t1.tv_usec) / 1e6);
  printf("Export done in %lf s in file %s\n", duration, output_file);

  // ==================== CLEANUP ====================
  free(all_processed_regions);

  for (int w = 0; w < world_size; w++) {
    free(worker_regions[w]);
  }
  free(worker_regions);
  free(worker_region_count);
  free(worker_region_capacity);
}
