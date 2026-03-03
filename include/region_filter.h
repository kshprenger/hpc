#ifndef REGION_FILTER_H
#define REGION_FILTER_H

#include "gif_math.h"
#include "gif_model.h"
#include "split.h"
#include <math.h>
#include <mpi.h>
#include <omp.h>
#include <stdlib.h>

#define GHOST_WIDTH 5

// Apply gray filter to a single region
static inline void apply_gray_filter_to_region(Region *region) {
  if (!region || !region->p) {
    return;
  }

  int total_pixels = region->region_width * region->region_height;

#pragma omp parallel for schedule(static)
  for (int j = 0; j < total_pixels; j++) {
    int moy = (region->p[j].r + region->p[j].g + region->p[j].b) / 3;
    if (moy < 0)
      moy = 0;
    if (moy > 255)
      moy = 255;

    region->p[j].r = moy;
    region->p[j].g = moy;
    region->p[j].b = moy;
  }
}

// Perform one blur iteration
static inline int blur_iteration(Region *region, pixel *new_pixels, int size,
                                 int threshold) {
  int width = region->region_width;
  int height = region->region_height;
  pixel *p = region->p;
  const int denom = (2 * size + 1) * (2 * size + 1);

#pragma omp parallel for collapse(2) schedule(static)
  for (int j = 0; j < height; j++) {
    for (int k = 0; k < width; k++) {
      new_pixels[CONV(j, k, width)] = p[CONV(j, k, width)];
    }
  }

  int x0 = GHOST_WIDTH;
  int x1 = width - GHOST_WIDTH;

  int blur_x0 = (x0 > size) ? x0 : size;
  int blur_x1 = (x1 < width - size) ? x1 : (width - size);

#pragma omp parallel for collapse(2) schedule(static)
  for (int j = size; j < height / 10 - size; j++) {
    for (int k = blur_x0; k < blur_x1; k++) {
      int t_r = 0, t_g = 0, t_b = 0;

      for (int stencil_j = -size; stencil_j <= size; stencil_j++) {
        for (int stencil_k = -size; stencil_k <= size; stencil_k++) {
          pixel q = p[CONV(j + stencil_j, k + stencil_k, width)];
          t_r += q.r;
          t_g += q.g;
          t_b += q.b;
        }
      }

      int idx = CONV(j, k, width);
      new_pixels[idx].r = (unsigned char)(t_r / denom);
      new_pixels[idx].g = (unsigned char)(t_g / denom);
      new_pixels[idx].b = (unsigned char)(t_b / denom);
    }
  }

#pragma omp parallel for collapse(2) schedule(static)
  for (int j = (int)(height * 0.9) + size; j < height - size; j++) {
    for (int k = blur_x0; k < blur_x1; k++) {
      int t_r = 0, t_g = 0, t_b = 0;

      for (int stencil_j = -size; stencil_j <= size; stencil_j++) {
        for (int stencil_k = -size; stencil_k <= size; stencil_k++) {
          pixel q = p[CONV(j + stencil_j, k + stencil_k, width)];
          t_r += q.r;
          t_g += q.g;
          t_b += q.b;
        }
      }

      int idx = CONV(j, k, width);
      new_pixels[idx].r = (unsigned char)(t_r / denom);
      new_pixels[idx].g = (unsigned char)(t_g / denom);
      new_pixels[idx].b = (unsigned char)(t_b / denom);
    }
  }

  int local_end = 1;

#pragma omp parallel for collapse(2) reduction(&& : local_end) schedule(static)
  for (int j = 1; j < height - 1; j++) {
    for (int k = x0; k < x1; k++) {
      int idx = CONV(j, k, width);

      float diff_r = (float)new_pixels[idx].r - (float)p[idx].r;
      float diff_g = (float)new_pixels[idx].g - (float)p[idx].g;
      float diff_b = (float)new_pixels[idx].b - (float)p[idx].b;

      int ok =
          !(diff_r > threshold || -diff_r > threshold || diff_g > threshold ||
            -diff_g > threshold || diff_b > threshold || -diff_b > threshold);

      local_end = local_end && ok;
    }
  }

#pragma omp parallel for collapse(2) schedule(static)
  for (int j = 1; j < height - 1; j++) {
    for (int k = x0; k < x1; k++) {
      p[CONV(j, k, width)] = new_pixels[CONV(j, k, width)];
    }
  }

  return local_end;
}

// Exchange ghost cells with neighboring workers
static inline void exchange_ghost_cells(Region *region, MPI_Comm comm) {
  int region_id = region->region_id;
  int k_regions = region->k_regions;
  int width = region->region_width;
  int height = region->region_height;
  pixel *p = region->p;

  int ghost_size = GHOST_WIDTH * height;

  pixel *send_left = NULL;
  pixel *send_right = NULL;
  pixel *recv_left = NULL;
  pixel *recv_right = NULL;

  MPI_Request requests[4];
  int num_requests = 0;

  // Do we have left and right neighbours?
  int has_left_neighbor = (region_id > 0);
  int has_right_neighbor = (region_id < k_regions - 1);

  if (has_left_neighbor) {
    send_left = (pixel *)malloc(ghost_size * sizeof(pixel));
    recv_left = (pixel *)malloc(ghost_size * sizeof(pixel));

    for (int y = 0; y < height; y++) {
      for (int x = 0; x < GHOST_WIDTH; x++) {
        send_left[y * GHOST_WIDTH + x] = p[CONV(y, GHOST_WIDTH + x, width)];
      }
    }

    // Async makes it faster!
    MPI_Isend(send_left, ghost_size * sizeof(pixel), MPI_BYTE, region_id - 1, 0,
              comm, &requests[num_requests++]);
    MPI_Irecv(recv_left, ghost_size * sizeof(pixel), MPI_BYTE, region_id - 1, 1,
              comm, &requests[num_requests++]);
  }

  if (has_right_neighbor) {
    send_right = (pixel *)malloc(ghost_size * sizeof(pixel));
    recv_right = (pixel *)malloc(ghost_size * sizeof(pixel));

    for (int y = 0; y < height; y++) {
      for (int x = 0; x < GHOST_WIDTH; x++) {
        send_right[y * GHOST_WIDTH + x] =
            p[CONV(y, width - 2 * GHOST_WIDTH + x, width)];
      }
    }

    // Async makes it faster!
    MPI_Isend(send_right, ghost_size * sizeof(pixel), MPI_BYTE, region_id + 1,
              1, comm, &requests[num_requests++]);
    MPI_Irecv(recv_right, ghost_size * sizeof(pixel), MPI_BYTE, region_id + 1,
              0, comm, &requests[num_requests++]);
  }

  // Await exchanges
  if (num_requests > 0) {
    MPI_Waitall(num_requests, requests, MPI_STATUSES_IGNORE);
  }

  if (has_left_neighbor && recv_left) {
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < GHOST_WIDTH; x++) {
        p[CONV(y, x, width)] = recv_left[y * GHOST_WIDTH + x];
      }
    }
    free(recv_left);
    free(send_left);
  }

  if (has_right_neighbor && recv_right) {
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < GHOST_WIDTH; x++) {
        p[CONV(y, width - GHOST_WIDTH + x, width)] =
            recv_right[y * GHOST_WIDTH + x];
      }
    }
    free(recv_right);
    free(send_right);
  }
}

static inline void apply_blur_filter_to_region_mpi(Region *region, int size,
                                                   int threshold,
                                                   MPI_Comm comm) {
  if (!region || !region->p) {
    return;
  }

  int width = region->region_width;
  int height = region->region_height;
  int k_regions = region->k_regions;

  pixel *new_pixels = (pixel *)malloc(width * height * sizeof(pixel));
  if (!new_pixels) {
    return;
  }

  int global_end = 0;

  do {
    int local_end = blur_iteration(region, new_pixels, size, threshold);

    // If image is split across multiple workers, sync ghost cells and
    // convergence
    if (k_regions > 1) {
      exchange_ghost_cells(region, comm);

      MPI_Allreduce(&local_end, &global_end, 1, MPI_INT, MPI_LAND, comm);
    } else {
      global_end = local_end;
    }

  } while (threshold > 0 && !global_end);

  free(new_pixels);
}

static inline void apply_blur_filter_to_region(Region *region, int size,
                                               int threshold) {
  if (!region || !region->p) {
    return;
  }

  int width = region->region_width;
  int height = region->region_height;

  pixel *new_pixels = (pixel *)malloc(width * height * sizeof(pixel));
  if (!new_pixels) {
    return;
  }

  int end = 0;

  do {
    end = blur_iteration(region, new_pixels, size, threshold);
  } while (threshold > 0 && !end);

  free(new_pixels);
}

static inline void apply_sobel_filter_to_region(Region *region) {
  if (!region || !region->p) {
    return;
  }

  int width = region->region_width;
  int height = region->region_height;
  pixel *p = region->p;

  pixel *sobel = (pixel *)malloc(width * height * sizeof(pixel));
  if (!sobel) {
    return;
  }

#pragma omp parallel for collapse(2) schedule(static)
  for (int j = 1; j < height - 1; j++) {
    for (int k = 1; k < width - 1; k++) {
      int pixel_blue_no = p[CONV(j - 1, k - 1, width)].b;
      int pixel_blue_n = p[CONV(j - 1, k, width)].b;
      int pixel_blue_ne = p[CONV(j - 1, k + 1, width)].b;
      int pixel_blue_so = p[CONV(j + 1, k - 1, width)].b;
      int pixel_blue_s = p[CONV(j + 1, k, width)].b;
      int pixel_blue_se = p[CONV(j + 1, k + 1, width)].b;
      int pixel_blue_o = p[CONV(j, k - 1, width)].b;
      int pixel_blue_e = p[CONV(j, k + 1, width)].b;

      float deltaX_blue = -pixel_blue_no + pixel_blue_ne - 2.0f * pixel_blue_o +
                          2.0f * pixel_blue_e - pixel_blue_so + pixel_blue_se;

      float deltaY_blue = pixel_blue_se + 2.0f * pixel_blue_s + pixel_blue_so -
                          pixel_blue_ne - 2.0f * pixel_blue_n - pixel_blue_no;

      float val_blue =
          sqrtf(deltaX_blue * deltaX_blue + deltaY_blue * deltaY_blue) / 4.0f;

      pixel out =
          (val_blue > 50.0f) ? (pixel){255, 255, 255} : (pixel){0, 0, 0};
      sobel[CONV(j, k, width)] = out;
    }
  }

  int x0 = GHOST_WIDTH;
  int x1 = width - GHOST_WIDTH;

#pragma omp parallel for collapse(2) schedule(static)
  for (int j = 1; j < height - 1; j++) {
    for (int k = x0; k < x1; k++) {
      p[CONV(j, k, width)] = sobel[CONV(j, k, width)];
    }
  }

  free(sobel);
}

static inline void apply_all_filters_to_region(Region *region, int blur_size,
                                               int blur_threshold) {
  apply_gray_filter_to_region(region);
  apply_blur_filter_to_region(region, blur_size, blur_threshold);
  apply_sobel_filter_to_region(region);
}

static inline void apply_all_filters_to_region_mpi(Region *region,
                                                   int blur_size,
                                                   int blur_threshold,
                                                   MPI_Comm comm) {
  apply_gray_filter_to_region(region);
  apply_blur_filter_to_region_mpi(region, blur_size, blur_threshold, comm);
  apply_sobel_filter_to_region(region);
}

#endif
