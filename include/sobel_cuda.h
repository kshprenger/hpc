#ifndef SOBEL_CUDA_H
#define SOBEL_CUDA_H

#include <stdio.h>

#include "filter_api.h"
#include "gif_math.h"
#include "gif_model.h"
#include "region_filter.h"

#ifdef USE_CUDA

extern int cuda_is_available(void);
extern void cuda_get_device_name(char *name, int max_len);

/* Legacy whole-image sobel (non-region path) */
extern void apply_sobel_filter_cuda(animated_gif *image);

/* Per-region CUDA filter functions */
extern void apply_gray_filter_to_region_cuda(pixel *region_pixels, int width,
                                             int height);

extern void apply_blur_filter_to_region_cuda(pixel *region_pixels, int width,
                                             int height, int size,
                                             int threshold, int region_id,
                                             int k_regions);

extern void apply_sobel_filter_to_region_cuda(pixel *region_pixels, int width,
                                              int height, int ghost_width,
                                              int region_id, int k_regions);

#endif /* USE_CUDA */

static inline int has_nvidia_gpu(void) {
#ifdef USE_CUDA
  return cuda_is_available();
#else
  return 0;
#endif
}

static inline void print_gpu_info(int mpi_rank) {
#ifdef USE_CUDA
  if (cuda_is_available()) {
    char device_name[256];
    cuda_get_device_name(device_name, sizeof(device_name));
    printf("[Rank %d] Using CUDA GPU: %s\n", mpi_rank, device_name);
  } else {
    printf("[Rank %d] No CUDA GPU available, using CPU\n", mpi_rank);
  }
#else
  printf("[Rank %d] CUDA support not compiled in, using CPU\n", mpi_rank);
#endif
}

static inline void apply_sobel_filter_dispatch(animated_gif *image,
                                               int use_gpu) {
#ifdef USE_CUDA
  if (use_gpu && cuda_is_available()) {
    apply_sobel_filter_cuda(image);
  } else {
    apply_sobel_filter(image);
  }
#else
  (void)use_gpu;
  apply_sobel_filter(image);
#endif
}

static inline void apply_gray_filter_to_region_dispatch(Region *region,
                                                        int use_gpu) {
  if (!region || !region->p)
    return;

#ifdef USE_CUDA
  if (use_gpu && cuda_is_available()) {
    apply_gray_filter_to_region_cuda(region->p, region->region_width,
                                     region->region_height);
    return;
  }
#else
  (void)use_gpu;
#endif
  apply_gray_filter_to_region(region);
}

static inline void apply_blur_filter_to_region_dispatch(Region *region,
                                                        int size, int threshold,
                                                        int use_gpu) {
  if (!region || !region->p)
    return;

#ifdef USE_CUDA
  if (use_gpu && cuda_is_available()) {
    apply_blur_filter_to_region_cuda(region->p, region->region_width,
                                     region->region_height, size, threshold,
                                     region->region_id, region->k_regions);
    return;
  }
#else
  (void)use_gpu;
#endif
  apply_blur_filter_to_region(region, size, threshold);
}

static inline void apply_blur_filter_to_region_mpi_dispatch(
    Region *region, int size, int threshold, MPI_Comm comm, int use_gpu) {
  if (!region || !region->p)
    return;

  /*
   * The CUDA blur path handles convergence internally and does not require
   * MPI ghost-cell synchronisation because it operates on the full bordered
   * region buffer that already contains ghost pixels from the Split step.
   * For the multi-region (split) case we therefore still fall back to the
   * CPU MPI path which exchanges ghost cells every iteration, ensuring
   * correctness.  Single-region images can safely use CUDA.
   */
#ifdef USE_CUDA
  if (use_gpu && cuda_is_available() && region->k_regions == 1) {
    apply_blur_filter_to_region_cuda(region->p, region->region_width,
                                     region->region_height, size, threshold,
                                     region->region_id, region->k_regions);
    return;
  }
#else
  (void)use_gpu;
#endif
  apply_blur_filter_to_region_mpi(region, size, threshold, comm);
}

static inline void apply_sobel_filter_to_region_dispatch(Region *region,
                                                         int use_gpu) {
  if (!region || !region->p)
    return;

#ifdef USE_CUDA
  if (use_gpu && cuda_is_available()) {
    apply_sobel_filter_to_region_cuda(region->p, region->region_width,
                                      region->region_height, GHOST_WIDTH,
                                      region->region_id, region->k_regions);
    return;
  }
#else
  (void)use_gpu;
#endif
  apply_sobel_filter_to_region(region);
}

static inline void apply_all_filters_to_region_gpu(Region *region,
                                                   int blur_size,
                                                   int blur_threshold,
                                                   int use_gpu) {
  apply_gray_filter_to_region_dispatch(region, use_gpu);
  apply_blur_filter_to_region_dispatch(region, blur_size, blur_threshold,
                                       use_gpu);
  apply_sobel_filter_to_region_dispatch(region, use_gpu);
}

static inline void apply_all_filters_to_region_mpi_gpu(Region *region,
                                                       int blur_size,
                                                       int blur_threshold,
                                                       MPI_Comm comm,
                                                       int use_gpu) {
  apply_gray_filter_to_region_dispatch(region, use_gpu);
  apply_blur_filter_to_region_mpi_dispatch(region, blur_size, blur_threshold,
                                           comm, use_gpu);
  apply_sobel_filter_to_region_dispatch(region, use_gpu);
}

#endif /* SOBEL_CUDA_H */
