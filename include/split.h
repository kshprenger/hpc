#ifndef TASK_H
#define TASK_H
#include "gif_model.h"
#include <assert.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdlib.h>

// Represents smallest task which distributes over MPI and is being feeded to
// openmp runtime: either whole image or part of splitting.
typedef struct Region {
  int region_id; // For distinguishing from regions of other images
  int width;
  int height;
  pixel *p;
  bool whole;
} Region;

// Splits image to k^2 Regions and adds extra borders for each region to sobel
// filter. Each region will have overlapping 1-pixel borders from the original
// image so Sobel filter can be applied independently to each region.
// Other filter can just use inner part of region excluding borders.
//
// Why k^2 parts? k^2 part splitting always results into equal
// parts. (event if width%k or height%k !=0, last region will be slightly
// bigger by eating reminder from this division)
//
// For example if we want to process 5 images on 4 ranks - one image will be
// splitted using k=2 to 4 pieces.
Region *Split(pixel *p, int image_id, int width, int height, int k) {
  if (!p || k == 0) {
    assert(false);
  }

  // Calculate base dimensions for each region (without borders)
  int region_width = width / k;
  int region_height = height / k;

  Region *regions = (Region *)malloc(k * k * sizeof(Region));
  if (!regions) {
    assert(false);
  }

  for (size_t row = 0; row < k; row++) {
    for (size_t col = 0; col < k; col++) {
      size_t region_idx = row * k + col;

      // Calculate region boundaries (without borders)
      int start_x = col * region_width;
      int start_y = row * region_height;
      int end_x = (col == k - 1) ? width : start_x + region_width;
      int end_y = (row == k - 1) ? height : start_y + region_height;

      // Add 1-pixel border for Sobel filter
      int border_start_x = (start_x > 0) ? start_x - 1 : 0;
      int border_start_y = (start_y > 0) ? start_y - 1 : 0;
      int border_end_x = (end_x < width) ? end_x + 1 : width;
      int border_end_y = (end_y < height) ? end_y + 1 : height;

      // Calculate dimensions with borders
      int bordered_width = border_end_x - border_start_x;
      int bordered_height = border_end_y - border_start_y;

      regions[region_idx].width = bordered_width;
      regions[region_idx].height = bordered_height;
      regions[region_idx].whole = (k == 1);
      regions[region_idx].region_id = image_id;

      regions[region_idx].p =
          (pixel *)malloc(bordered_width * bordered_height * sizeof(pixel));
      if (!regions[region_idx].p) {
        assert(false);
      }

      // Copy pixel data with borders
      for (int y = 0; y < bordered_height; y++) {
        for (int x = 0; x < bordered_width; x++) {
          int src_x = border_start_x + x;
          int src_y = border_start_y + y;
          int src_idx = src_y * width + src_x;
          int dst_idx = y * bordered_width + x;

          regions[region_idx].p[dst_idx] = p[src_idx];
        }
      }
    }
  }

  return regions;
}

// Combines k^2 Regions back into a single image
pixel *Combine(Region *regions, int width, int height, int k) {
  if (!regions || k == 0) {
    assert(false);
  }

  pixel *result = (pixel *)malloc(width * height * sizeof(pixel));
  if (!result) {
    assert(false);
  }

  // Calculate base region dimensions
  int region_width = width / k;
  int region_height = height / k;

  // Copy processed data back from regions (excluding extra borders from sobel
  // filter)
  for (int row = 0; row < k; row++) {
    for (int col = 0; col < k; col++) {
      size_t region_idx = row * k + col;
      Region *region = &regions[region_idx];

      // Calculate original region boundaries
      int start_x = col * region_width;
      int start_y = row * region_height;
      int end_x = (col == k - 1) ? width : start_x + region_width;
      int end_y = (row == k - 1) ? height : start_y + region_height;

      // Calculate border offset (how much border was added)
      int border_offset_x = (start_x > 0) ? 1 : 0;
      int border_offset_y = (start_y > 0) ? 1 : 0;

      // Copy pixel data back (excluding borders)
      for (int y = start_y; y < end_y; y++) {
        for (int x = start_x; x < end_x; x++) {
          int src_x = x - start_x + border_offset_x;
          int src_y = y - start_y + border_offset_y;
          int src_idx = src_y * region->width + src_x;
          int dst_idx = y * width + x;

          result[dst_idx] = region->p[src_idx];
        }
      }
    }
  }

  return result;
}

void FreeRegions(Region *regions, size_t k) {
  if (!regions) {
    assert(false);
  }

  for (size_t i = 0; i < k * k; i++) {
    if (regions[i].p) {
      free(regions[i].p);
    }
  }
  free(regions);
}

#endif
