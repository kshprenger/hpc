#include "gif_model.h"
#include "persist_api.h"
#include "region_filter.h"
#include "split.h"
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#define TAG_COMMAND 0
#define TAG_REGION_COUNT 1
#define TAG_BUFFER_SIZE 2
#define TAG_BUFFER_DATA 3
#define TAG_RESULT_SIZE 4
#define TAG_RESULT_DATA 5

// Master -> Workers
#define CMD_PROCESS_SPLIT_IMAGE 1
#define CMD_PROCESS_BATCH 2
#define CMD_TERMINATE 3

static int compare_regions(const void *a, const void *b) {
  const Region *ra = (const Region *)a;
  const Region *rb = (const Region *)b;
  if (ra->image_id != rb->image_id) {
    return ra->image_id - rb->image_id;
  }
  return ra->region_id - rb->region_id;
}

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

static void process_split_image(animated_gif *image, int image_idx,
                                int world_size, Region **result_regions) {
  Region *regions =
      Split(image->p[image_idx], image_idx, image->width[image_idx],
            image->height[image_idx], world_size);

  int cmd = CMD_PROCESS_SPLIT_IMAGE;
  for (int w = 1; w < world_size; w++) {
    MPI_Send(&cmd, 1, MPI_INT, w, TAG_COMMAND, MPI_COMM_WORLD);
  }

  for (int w = 1; w < world_size; w++) {
    int buffer_size = calculate_batch_buffer_size(&regions[w], 1);
    char *buffer = (char *)malloc(buffer_size);
    pack_regions(&regions[w], 1, buffer, buffer_size, MPI_COMM_WORLD);

    MPI_Send(&buffer_size, 1, MPI_INT, w, TAG_BUFFER_SIZE, MPI_COMM_WORLD);
    MPI_Send(buffer, buffer_size, MPI_PACKED, w, TAG_BUFFER_DATA,
             MPI_COMM_WORLD);

    free(buffer);
    free(regions[w].p);
  }

  Region *master_region = &regions[0];

  apply_all_filters_to_region_mpi(master_region, 5, 20, MPI_COMM_WORLD);

  *result_regions = (Region *)malloc(world_size * sizeof(Region));
  (*result_regions)[0] = *master_region;

  for (int w = 1; w < world_size; w++) {
    int buffer_size;
    MPI_Recv(&buffer_size, 1, MPI_INT, w, TAG_RESULT_SIZE, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);

    char *buffer = (char *)malloc(buffer_size);
    MPI_Recv(buffer, buffer_size, MPI_PACKED, w, TAG_RESULT_DATA,
             MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    unpack_regions(&(*result_regions)[w], 1, buffer, buffer_size,
                   MPI_COMM_WORLD);
    free(buffer);
  }

  free(regions);
}

static void process_nonsplit_images_batch(animated_gif *image,
                                          int *image_indices, int num_images,
                                          int world_size,
                                          Region **result_regions,
                                          int *result_count) {
  Region *all_regions = (Region *)malloc(num_images * sizeof(Region));
  for (int i = 0; i < num_images; i++) {
    int idx = image_indices[i];
    Region *r =
        Split(image->p[idx], idx, image->width[idx], image->height[idx], 1);
    all_regions[i] = r[0];
    free(r);
  }

  Region **worker_regions = (Region **)malloc(world_size * sizeof(Region *));
  int *worker_counts = (int *)calloc(world_size, sizeof(int));

  for (int w = 0; w < world_size; w++) {
    worker_regions[w] = (Region *)malloc(num_images * sizeof(Region));
  }

  for (int i = 0; i < num_images; i++) {
    int worker_id = i % world_size;
    worker_regions[worker_id][worker_counts[worker_id]++] = all_regions[i];
  }

  free(all_regions);

  int cmd = CMD_PROCESS_BATCH;
  for (int w = 1; w < world_size; w++) {
    MPI_Send(&cmd, 1, MPI_INT, w, TAG_COMMAND, MPI_COMM_WORLD);
  }

  for (int w = 1; w < world_size; w++) {
    int count = worker_counts[w];
    MPI_Send(&count, 1, MPI_INT, w, TAG_REGION_COUNT, MPI_COMM_WORLD);

    if (count > 0) {
      int buffer_size = calculate_batch_buffer_size(worker_regions[w], count);
      char *buffer = (char *)malloc(buffer_size);
      pack_regions(worker_regions[w], count, buffer, buffer_size,
                   MPI_COMM_WORLD);

      MPI_Send(&buffer_size, 1, MPI_INT, w, TAG_BUFFER_SIZE, MPI_COMM_WORLD);
      MPI_Send(buffer, buffer_size, MPI_PACKED, w, TAG_BUFFER_DATA,
               MPI_COMM_WORLD);

      free(buffer);

      for (int r = 0; r < count; r++) {
        free(worker_regions[w][r].p);
      }
    }
  }

  int master_count = worker_counts[0];
  for (int r = 0; r < master_count; r++) {
    apply_all_filters_to_region(&worker_regions[0][r], 5, 20);
  }

  *result_count = num_images;
  *result_regions = (Region *)malloc(num_images * sizeof(Region));
  int result_idx = 0;

  for (int r = 0; r < master_count; r++) {
    (*result_regions)[result_idx++] = worker_regions[0][r];
  }

  for (int w = 1; w < world_size; w++) {
    int count = worker_counts[w];
    if (count > 0) {
      int buffer_size;
      MPI_Recv(&buffer_size, 1, MPI_INT, w, TAG_RESULT_SIZE, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);

      char *buffer = (char *)malloc(buffer_size);
      MPI_Recv(buffer, buffer_size, MPI_PACKED, w, TAG_RESULT_DATA,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      unpack_regions(&(*result_regions)[result_idx], count, buffer, buffer_size,
                     MPI_COMM_WORLD);
      free(buffer);
      result_idx += count;
    }
  }

  for (int w = 0; w < world_size; w++) {
    free(worker_regions[w]);
  }
  free(worker_regions);
  free(worker_counts);
}

void Master(char *input_file, char *output_file) {
  int rank, world_size;
  animated_gif *image = NULL;
  struct timeval t1, t2;
  double duration;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

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

  // Special case: Single rank - no MPI communication
  if (world_size == 1) {
    gettimeofday(&t1, NULL);

    for (int i = 0; i < n_images; i++) {
      Region *regions =
          Split(image->p[i], i, image->width[i], image->height[i], 1);
      apply_all_filters_to_region(&regions[0], 5, 20);
      pixel *combined = Combine(regions, image->width[i], image->height[i], 1);
      free(image->p[i]);
      image->p[i] = combined;
      free(regions[0].p);
      free(regions);
    }

    gettimeofday(&t2, NULL);
    duration = (t2.tv_sec - t1.tv_sec) + ((t2.tv_usec - t1.tv_usec) / 1e6);
    printf("SOBEL done in %lf s\n", duration);

    gettimeofday(&t1, NULL);

    if (!store_pixels(output_file, image)) {
      fprintf(stderr, "Master: Failed to store GIF to %s\n", output_file);
      return;
    }

    return;
  }

  // Multi-rank processing
  gettimeofday(&t1, NULL);

  int *split_images = (int *)malloc(n_images * sizeof(int));
  int *nonsplit_images = (int *)malloc(n_images * sizeof(int));
  int num_split = 0;
  int num_nonsplit = 0;

  if (n_images < world_size) {
    for (int i = 0; i < n_images; i++) {
      split_images[num_split++] = i;
    }
  } else {
    int remainder = n_images % world_size;
    for (int i = 0; i < remainder; i++) {
      split_images[num_split++] = i;
    }
    for (int i = remainder; i < n_images; i++) {
      nonsplit_images[num_nonsplit++] = i;
    }
  }

  Region *all_results =
      (Region *)malloc((n_images * world_size) * sizeof(Region));
  int total_results = 0;

  // Process spliTted images one at a time (requires ghost cell sync)
  // Parallelize???
  for (int i = 0; i < num_split; i++) {
    Region *split_results;
    process_split_image(image, split_images[i], world_size, &split_results);

    for (int r = 0; r < world_size; r++) {
      all_results[total_results++] = split_results[r];
    }
    free(split_results);
  }

  // Process non-splitted images as a batch (no ghost cell sync needed)
  if (num_nonsplit > 0) {
    Region *batch_results;
    int batch_count;
    process_nonsplit_images_batch(image, nonsplit_images, num_nonsplit,
                                  world_size, &batch_results, &batch_count);

    for (int r = 0; r < batch_count; r++) {
      all_results[total_results++] = batch_results[r];
    }
    free(batch_results);
  }

  int cmd = CMD_TERMINATE;
  for (int w = 1; w < world_size; w++) {
    MPI_Send(&cmd, 1, MPI_INT, w, TAG_COMMAND, MPI_COMM_WORLD);
  }

  gettimeofday(&t2, NULL);
  duration = (t2.tv_sec - t1.tv_sec) + ((t2.tv_usec - t1.tv_usec) / 1e6);
  printf("SOBEL done in %lf s\n", duration);

  gettimeofday(&t1, NULL);

  // Sort results by image_id then  by region_id
  qsort(all_results, total_results, sizeof(Region), compare_regions);

  int region_idx = 0;
  for (int i = 0; i < n_images; i++) {
    int start_idx = region_idx;
    int image_region_count = 0;

    while (region_idx < total_results &&
           all_results[region_idx].image_id == i) {
      image_region_count++;
      region_idx++;
    }

    if (image_region_count == 0) {
      fprintf(stderr, "Master: No regions found for image %d\n", i);
      continue;
    }

    int k_regions = all_results[start_idx].k_regions;
    pixel *combined = Combine(&all_results[start_idx], image->width[i],
                              image->height[i], k_regions);

    free(image->p[i]);
    image->p[i] = combined;

    for (int r = start_idx; r < start_idx + image_region_count; r++) {
      if (all_results[r].p) {
        free(all_results[r].p);
        all_results[r].p = NULL;
      }
    }
  }

  if (!store_pixels(output_file, image)) {
    fprintf(stderr, "Master: Failed to store GIF to %s\n", output_file);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
}
