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
                                               int use_gpu, runtime_config_t config) {
#ifdef USE_CUDA
  if (use_gpu && cuda_is_available() && config.cuda_mode != CUDA_MODE_OFF) {
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
                                                        int use_gpu, runtime_config_t config) {
  if (!region || !region->p)
    return;

#ifdef USE_CUDA
  if (use_gpu && cuda_is_available() && config.cuda_mode != CUDA_MODE_OFF) {
    apply_gray_filter_to_region_cuda(region->p, region->region_width,
                                     region->region_height);
    return;
  }
#else
  (void)use_gpu;
#endif
  openmp_mode_t sg_openmp_mode = config.openmp_mode;
  int num_threads = omp_get_max_threads();
  if (config.openmp_mode == OPENMP_MODE_AUTO ) {
    if (region->region_height * region->region_width > OPENMP_THRESHOLD 
      && num_threads > OPENMP_THREADS_THRESHOLD) {
          sg_openmp_mode = OPENMP_MODE_FORCE;
    } else {
          sg_openmp_mode = OPENMP_MODE_OFF;
    }
  }

  if (sg_openmp_mode != OPENMP_MODE_OFF) {
    printf("Applying gray to region %d of image %d with OpenMP parallelization using %d threads\n",
           region->region_id, region->image_id, num_threads);
  }
  else {
    printf("Applying gray to region %d of image %d without OpenMP parallelization\n",
           region->region_id, region->image_id);
  }
  apply_gray_filter_to_region(region, sg_openmp_mode);
}

static inline void apply_blur_filter_to_region_dispatch(Region *region,
                                                        int size, int threshold,
                                                        int use_gpu, runtime_config_t config) {
  if (!region || !region->p)
    return;

#ifdef USE_CUDA
  if (use_gpu && cuda_is_available() && config.cuda_mode != CUDA_MODE_OFF) {
    apply_blur_filter_to_region_cuda(region->p, region->region_width,
                                     region->region_height, size, threshold,
                                     region->region_id, region->k_regions);
    return;
  }
#else
  (void)use_gpu;
#endif
  openmp_mode_t sg_openmp_mode = config.openmp_mode;
  int num_threads = omp_get_max_threads();
  if (config.openmp_mode == OPENMP_MODE_AUTO ) {
    if (region->region_height * region->region_width > OPENMP_THRESHOLD 
      && num_threads > OPENMP_THREADS_THRESHOLD) {
          sg_openmp_mode = OPENMP_MODE_FORCE;
    } else {
          sg_openmp_mode = OPENMP_MODE_OFF;
    }
  }

  if (sg_openmp_mode != OPENMP_MODE_OFF) {
    printf("Applying blur to region %d of image %d with OpenMP parallelization using %d threads\n",
           region->region_id, region->image_id, num_threads);
  }
  else {
    printf("Applying blur to region %d of image %d without OpenMP parallelization\n",
           region->region_id, region->image_id);
  }
  apply_blur_filter_to_region(region, size, threshold, sg_openmp_mode);
}

static inline void apply_blur_filter_to_region_mpi_dispatch(
    Region *region, int size, int threshold, MPI_Comm comm, int use_gpu, runtime_config_t config) {
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
  if (use_gpu && cuda_is_available() && region->k_regions == 1 && config.cuda_mode != CUDA_MODE_OFF) {
    apply_blur_filter_to_region_cuda(region->p, region->region_width,
                                     region->region_height, size, threshold,
                                     region->region_id, region->k_regions);
    return;
  }
  else if (region->k_regions > 1) {
    printf("Region %d of image %d is part of a split image, skipping GPU acceleration for blur filter\n",
           region->region_id, region->image_id, region->k_regions);
  }
#else
  (void)use_gpu;
#endif
  openmp_mode_t blur_openmp_mode = config.openmp_mode;
  if (config.openmp_mode == OPENMP_MODE_AUTO ) {
     int num_threads = omp_get_max_threads();
     if (region->region_height * region->region_width > GHOST_OPENMP_THRESHOLD && 
         num_threads > GHOST_OPENMP_THREADS_THRESHOLD) {
         blur_openmp_mode = OPENMP_MODE_FORCE;
     } else {
         blur_openmp_mode = OPENMP_MODE_OFF;
     }
  }
  if (blur_openmp_mode != OPENMP_MODE_OFF) {
    printf("Applying blur filter to region %d of image %d with OpenMP parallelization using %d threads\n",
           region->region_id, region->image_id, omp_get_max_threads());
  }
  else {
    printf("Applying blur filter to region %d of image %d without OpenMP parallelization\n",
           region->region_id, region->image_id);
  }
  apply_blur_filter_to_region_mpi(region, size, threshold, comm, blur_openmp_mode);
}

static inline void apply_sobel_filter_to_region_dispatch(Region *region,
                                                         int use_gpu, 
                                                         runtime_config_t config) {
  if (!region || !region->p)
    return;

#ifdef USE_CUDA
  if (use_gpu && cuda_is_available() && config.cuda_mode != CUDA_MODE_OFF) {
    apply_sobel_filter_to_region_cuda(region->p, region->region_width,
                                      region->region_height, GHOST_WIDTH,
                                      region->region_id, region->k_regions);
    return;
  }
#else
  (void)use_gpu;
#endif
  openmp_mode_t sg_openmp_mode = config.openmp_mode;
  int num_threads = omp_get_max_threads();
  if (config.openmp_mode == OPENMP_MODE_AUTO ) {
    if (region->region_height * region->region_width > OPENMP_THRESHOLD 
      && num_threads > OPENMP_THREADS_THRESHOLD) {
          sg_openmp_mode = OPENMP_MODE_FORCE;
    } else {
          sg_openmp_mode = OPENMP_MODE_OFF;
    }
  }

  if (sg_openmp_mode != OPENMP_MODE_OFF) {
    printf("Applying sobel to region %d of image %d with OpenMP parallelization using %d threads\n",
           region->region_id, region->image_id, num_threads);
  }
  else {
    printf("Applying sobel to region %d of image %d without OpenMP parallelization\n",
           region->region_id, region->image_id);
  }
  apply_sobel_filter_to_region(region, sg_openmp_mode);
}

static inline void apply_all_filters_to_region_gpu(Region *region,
                                                   int blur_size,
                                                   int blur_threshold,
                                                   int use_gpu, runtime_config_t config) {
  if (use_gpu && config.cuda_mode == CUDA_MODE_AUTO) {
    if (region->region_height * region->region_width > CUDA_THRESHOLD) {
      use_gpu = 1;
    } else {
      printf("Region %d of image %d is too small for GPU acceleration, using CPU\n",
             region->region_id, region->image_id);
      use_gpu = 0;
    }
  }
  if (config.cuda_mode == CUDA_MODE_OFF) {
    printf("Warning: Cuda disabled in runtime config, using CPU for all filters\n");
  }
  apply_gray_filter_to_region_dispatch(region, use_gpu, config);
  apply_blur_filter_to_region_dispatch(region, blur_size, blur_threshold,
                                       use_gpu, config);
  apply_sobel_filter_to_region_dispatch(region, use_gpu, config);
}

static inline void apply_all_filters_to_region_mpi_gpu(Region *region,
                                                       int blur_size,
                                                       int blur_threshold,
                                                       MPI_Comm comm,
                                                       int use_gpu, runtime_config_t config) {
  if (use_gpu && config.cuda_mode == CUDA_MODE_AUTO) {
    if (region->region_height * region->region_width > CUDA_THRESHOLD) {
      use_gpu = 1;
    } else {
      printf("Region %d of image %d is too small for GPU acceleration, using CPU\n",
             region->region_id, region->image_id);
      use_gpu = 0;
    }
  }
  if (config.cuda_mode == CUDA_MODE_OFF) {
    printf("Warning: Cuda disabled in runtime config, using CPU for all filters\n");
  }
  apply_gray_filter_to_region_dispatch(region, use_gpu, config);
  apply_blur_filter_to_region_mpi_dispatch(region, blur_size, blur_threshold,
                                           comm, use_gpu, config);
  apply_sobel_filter_to_region_dispatch(region, use_gpu, config);
}

#endif /* SOBEL_CUDA_H */
