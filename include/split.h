#ifndef SPLIT_H
#define SPLIT_H
#include "gif_model.h"
#include <_abort.h>
#include <assert.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdlib.h>

// Represents smallest task which distributes over MPI and is being feeded to
// openmp runtime: either whole image or part of splitting.
typedef struct Region {
  int image_id;  // For distinguishing from regions of other images
  int region_id; // For sorting inside one image
  int region_width;
  int region_height;
  int k_regions;
  pixel *p;
} Region;

Region *Split(pixel *p, int image_id, int image_width, int image_height,
              int k_regions) {
  if (!p || k_regions == 0) {
    abort();
  }

  // Calculate base dimensions for each region (without borders)
  int region_width = image_width / k_regions;
  assert(region_width > 0);
  int region_height = image_height;

  Region *regions = (Region *)malloc(k_regions * sizeof(Region));
  if (!regions) {
    abort();
  }

  for (size_t region = 0; region < k_regions; region++) {
    // Calculate region boundaries (x-axis, without borders)
    int start_x = region * region_width;
    int end_x =
        (region == k_regions - 1) ? image_width : start_x + region_width;

    // Add 5-pixel border for blur filter
    int border_start_x = (start_x > 0) ? start_x - 5 : 0;
    int border_end_x = (end_x < image_width) ? end_x + 5 : image_width;

    int bordered_width = border_end_x - border_start_x;

    regions[region].region_id = region;
    regions[region].image_id = image_id;
    regions[region].region_width = bordered_width;
    regions[region].region_height = image_height;
    regions[region].k_regions = k_regions;

    regions[region].p =
        (pixel *)malloc(bordered_width * image_height * sizeof(pixel));
    if (!regions[region].p) {
      abort();
    }

    // Copy pixel data with borders
    for (int y = 0; y < image_height; y++) {
      for (int x = 0; x < bordered_width; x++) {
        int src_x = border_start_x + x;
        int src_y = y;
        int src_idx = src_y * image_width + src_x;
        int dst_idx = y * bordered_width + x;

        regions[region].p[dst_idx] = p[src_idx];
      }
    }
  }

  return regions;
}

// Combines k Regions back into a single image
pixel *Combine(Region *regions, int image_width, int image_height,
               int k_regions) {
  if (!regions || k_regions == 0) {
    abort();
  }

  pixel *result = (pixel *)malloc(image_width * image_height * sizeof(pixel));
  if (!result) {
    abort();
  }

  // Calculate base region width (without borders)
  int base_region_width = image_width / k_regions;

  for (int i = 0; i < k_regions; i++) {
    Region *region = &regions[i];
    int region_id = region->region_id;

    // Calculate original region boundaries (without borders)
    int start_x = region_id * base_region_width;
    int end_x = (region_id == k_regions - 1) ? image_width
                                             : start_x + base_region_width;
    int start_y = 0;
    int end_y = image_height;

    // Calculate border offset (5 pixels were added in Split)
    int border_offset_x = (start_x > 0) ? 5 : 0;
    int border_offset_y = 0; // No vertical borders for column splitting

    // Copy pixel data back (excluding borders)
    for (int y = start_y; y < end_y; y++) {
      for (int x = start_x; x < end_x; x++) {
        int src_x = x - start_x + border_offset_x;
        int src_y = y - start_y + border_offset_y;
        int src_idx = src_y * region->region_width + src_x;
        int dst_idx = y * image_width + x;

        result[dst_idx] = region->p[src_idx];
      }
    }
  }

  return result;
}

#endif
