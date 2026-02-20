#include "gif_math.h"
#include "gif_model.h"
#include <math.h>
#include <omp.h>
#include <stdlib.h>

void apply_sobel_filter(animated_gif *image) {
  if (image->n_images == 0) {
    return;
  }

  pixel **p = image->p;
  int width = image->width[0];
  int height = image->height[0];

  for (int i = 0; i < image->n_images; i++) {
    pixel *sobel = (pixel *)malloc(width * height * sizeof(pixel));
    if (!sobel) {
      abort();
    }

    #pragma omp parallel for collapse(2) schedule(static)
    for (int j = 1; j < height - 1; j++) {
      for (int k = 1; k < width - 1; k++) {
        int pixel_blue_no = p[i][CONV(j - 1, k - 1, width)].b;
        int pixel_blue_n  = p[i][CONV(j - 1, k,     width)].b;
        int pixel_blue_ne = p[i][CONV(j - 1, k + 1, width)].b;

        int pixel_blue_so = p[i][CONV(j + 1, k - 1, width)].b;
        int pixel_blue_s  = p[i][CONV(j + 1, k,     width)].b;
        int pixel_blue_se = p[i][CONV(j + 1, k + 1, width)].b;

        int pixel_blue_o  = p[i][CONV(j,     k - 1, width)].b;
        int pixel_blue_e  = p[i][CONV(j,     k + 1, width)].b;

        float deltaX_blue =
            -pixel_blue_no + pixel_blue_ne
            - 2.0f * pixel_blue_o + 2.0f * pixel_blue_e
            - pixel_blue_so + pixel_blue_se;

        float deltaY_blue =
            pixel_blue_se + 2.0f * pixel_blue_s + pixel_blue_so
            - pixel_blue_ne - 2.0f * pixel_blue_n - pixel_blue_no;

        float val_blue = sqrtf(deltaX_blue * deltaX_blue +
                               deltaY_blue * deltaY_blue) / 4.0f;

        pixel out;
        if (val_blue > 50.0f) {
          out.r = 255; out.g = 255; out.b = 255;
        } else {
          out.r = 0; out.g = 0; out.b = 0;
        }

        sobel[CONV(j, k, width)] = out;
      }
    }

    #pragma omp parallel for collapse(2) schedule(static)
    for (int j = 1; j < height - 1; j++) {
      for (int k = 1; k < width - 1; k++) {
        p[i][CONV(j, k, width)] = sobel[CONV(j, k, width)];
      }
    }

    free(sobel);
  }
}
