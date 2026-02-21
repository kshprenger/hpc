#ifndef REGION_FILTER_H
#define REGION_FILTER_H

#include "gif_math.h"
#include "gif_model.h"
#include "split.h"
#include <math.h>
#include <stdlib.h>

// Apply gray filter to a single region
static inline void apply_gray_filter_to_region(Region *region) {
  if (!region || !region->p) {
    return;
  }

  int total_pixels = region->region_width * region->region_height;
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

// Apply blur filter to a single region
static inline void apply_blur_filter_to_region(Region *region, int size,
                                               int threshold) {
  if (!region || !region->p) {
    return;
  }

  int width = region->region_width;
  int height = region->region_height;
  int end = 0;
  int n_iter = 0;

  pixel *p = region->p;
  pixel *new_pixels = (pixel *)malloc(width * height * sizeof(pixel));
  if (!new_pixels) {
    return;
  }

  do {
    end = 1;
    n_iter++;

    // Initialize new pixels with original values
    for (int j = 0; j < height - 1; j++) {
      for (int k = 0; k < width - 1; k++) {
        new_pixels[CONV(j, k, width)].r = p[CONV(j, k, width)].r;
        new_pixels[CONV(j, k, width)].g = p[CONV(j, k, width)].g;
        new_pixels[CONV(j, k, width)].b = p[CONV(j, k, width)].b;
      }
    }

    // Apply blur on top part of image (10%)
    for (int j = size; j < height / 10 - size; j++) {
      for (int k = size; k < width - size; k++) {
        int t_r = 0;
        int t_g = 0;
        int t_b = 0;

        for (int stencil_j = -size; stencil_j <= size; stencil_j++) {
          for (int stencil_k = -size; stencil_k <= size; stencil_k++) {
            t_r += p[CONV(j + stencil_j, k + stencil_k, width)].r;
            t_g += p[CONV(j + stencil_j, k + stencil_k, width)].g;
            t_b += p[CONV(j + stencil_j, k + stencil_k, width)].b;
          }
        }

        new_pixels[CONV(j, k, width)].r =
            t_r / ((2 * size + 1) * (2 * size + 1));
        new_pixels[CONV(j, k, width)].g =
            t_g / ((2 * size + 1) * (2 * size + 1));
        new_pixels[CONV(j, k, width)].b =
            t_b / ((2 * size + 1) * (2 * size + 1));
      }
    }

    // Copy the middle part of the image
    for (int j = height / 10 - size; j < height * 0.9 + size; j++) {
      for (int k = size; k < width - size; k++) {
        new_pixels[CONV(j, k, width)].r = p[CONV(j, k, width)].r;
        new_pixels[CONV(j, k, width)].g = p[CONV(j, k, width)].g;
        new_pixels[CONV(j, k, width)].b = p[CONV(j, k, width)].b;
      }
    }

    // Apply blur on the bottom part of the image (10%)
    for (int j = height * 0.9 + size; j < height - size; j++) {
      for (int k = size; k < width - size; k++) {
        int t_r = 0;
        int t_g = 0;
        int t_b = 0;

        for (int stencil_j = -size; stencil_j <= size; stencil_j++) {
          for (int stencil_k = -size; stencil_k <= size; stencil_k++) {
            t_r += p[CONV(j + stencil_j, k + stencil_k, width)].r;
            t_g += p[CONV(j + stencil_j, k + stencil_k, width)].g;
            t_b += p[CONV(j + stencil_j, k + stencil_k, width)].b;
          }
        }

        new_pixels[CONV(j, k, width)].r =
            t_r / ((2 * size + 1) * (2 * size + 1));
        new_pixels[CONV(j, k, width)].g =
            t_g / ((2 * size + 1) * (2 * size + 1));
        new_pixels[CONV(j, k, width)].b =
            t_b / ((2 * size + 1) * (2 * size + 1));
      }
    }

    // Check convergence and copy back
    for (int j = 1; j < height - 1; j++) {
      for (int k = 1; k < width - 1; k++) {
        float diff_r =
            (new_pixels[CONV(j, k, width)].r - p[CONV(j, k, width)].r);
        float diff_g =
            (new_pixels[CONV(j, k, width)].g - p[CONV(j, k, width)].g);
        float diff_b =
            (new_pixels[CONV(j, k, width)].b - p[CONV(j, k, width)].b);

        if (diff_r > threshold || -diff_r > threshold || diff_g > threshold ||
            -diff_g > threshold || diff_b > threshold || -diff_b > threshold) {
          end = 0;
        }

        p[CONV(j, k, width)].r = new_pixels[CONV(j, k, width)].r;
        p[CONV(j, k, width)].g = new_pixels[CONV(j, k, width)].g;
        p[CONV(j, k, width)].b = new_pixels[CONV(j, k, width)].b;
      }
    }

  } while (threshold > 0 && !end);

  free(new_pixels);
}

// Apply sobel filter to a single region
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

      float deltaX_blue = -pixel_blue_no + pixel_blue_ne - 2 * pixel_blue_o +
                          2 * pixel_blue_e - pixel_blue_so + pixel_blue_se;

      float deltaY_blue = pixel_blue_se + 2 * pixel_blue_s + pixel_blue_so -
                          pixel_blue_ne - 2 * pixel_blue_n - pixel_blue_no;

      float val_blue =
          sqrt(deltaX_blue * deltaX_blue + deltaY_blue * deltaY_blue) / 4;

      if (val_blue > 50) {
        sobel[CONV(j, k, width)].r = 255;
        sobel[CONV(j, k, width)].g = 255;
        sobel[CONV(j, k, width)].b = 255;
      } else {
        sobel[CONV(j, k, width)].r = 0;
        sobel[CONV(j, k, width)].g = 0;
        sobel[CONV(j, k, width)].b = 0;
      }
    }
  }

  for (int j = 1; j < height - 1; j++) {
    for (int k = 1; k < width - 1; k++) {
      p[CONV(j, k, width)].r = sobel[CONV(j, k, width)].r;
      p[CONV(j, k, width)].g = sobel[CONV(j, k, width)].g;
      p[CONV(j, k, width)].b = sobel[CONV(j, k, width)].b;
    }
  }

  free(sobel);
}

// Apply all filters to a region in sequence: gray -> blur -> sobel
static inline void apply_all_filters_to_region(Region *region, int blur_size,
                                               int blur_threshold) {
  apply_gray_filter_to_region(region);
  apply_blur_filter_to_region(region, blur_size, blur_threshold);
  apply_sobel_filter_to_region(region);
}

#endif
