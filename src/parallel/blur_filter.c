#include "gif_math.h"
#include "gif_model.h"
#include <stdlib.h>

void apply_blur_filter(animated_gif *image, int size, int threshold) {
  int i, j, k;
  int width, height;
  int end = 0;
  int n_iter = 0;

  pixel **p;
  pixel *new;

  /* Get the pixels of all images */
  p = image->p;

  /* Process all images */
  for (i = 0; i < image->n_images; i++) {
    n_iter = 0;
    width = image->width[i];
    height = image->height[i];

    /* Allocate array of new pixels */
    new = (pixel *)malloc(width * height * sizeof(pixel));

    /* Perform at least one blur iteration */
    do {
      end = 1;
      n_iter++;

      for (j = 0; j < height - 1; j++) {
        for (k = 0; k < width - 1; k++) {
          new[CONV(j, k, width)].r = p[i][CONV(j, k, width)].r;
          new[CONV(j, k, width)].g = p[i][CONV(j, k, width)].g;
          new[CONV(j, k, width)].b = p[i][CONV(j, k, width)].b;
        }
      }

      /* Apply blur on top part of image (10%) */
      for (j = size; j < height / 10 - size; j++) {
        for (k = size; k < width - size; k++) {
          int stencil_j, stencil_k;
          int t_r = 0;
          int t_g = 0;
          int t_b = 0;

          for (stencil_j = -size; stencil_j <= size; stencil_j++) {
            for (stencil_k = -size; stencil_k <= size; stencil_k++) {
              t_r += p[i][CONV(j + stencil_j, k + stencil_k, width)].r;
              t_g += p[i][CONV(j + stencil_j, k + stencil_k, width)].g;
              t_b += p[i][CONV(j + stencil_j, k + stencil_k, width)].b;
            }
          }

          new[CONV(j, k, width)].r = t_r / ((2 * size + 1) * (2 * size + 1));
          new[CONV(j, k, width)].g = t_g / ((2 * size + 1) * (2 * size + 1));
          new[CONV(j, k, width)].b = t_b / ((2 * size + 1) * (2 * size + 1));
        }
      }

      /* Copy the middle part of the image */
      for (j = height / 10 - size; j < height * 0.9 + size; j++) {
        for (k = size; k < width - size; k++) {
          new[CONV(j, k, width)].r = p[i][CONV(j, k, width)].r;
          new[CONV(j, k, width)].g = p[i][CONV(j, k, width)].g;
          new[CONV(j, k, width)].b = p[i][CONV(j, k, width)].b;
        }
      }

      /* Apply blur on the bottom part of the image (10%) */
      for (j = height * 0.9 + size; j < height - size; j++) {
        for (k = size; k < width - size; k++) {
          int stencil_j, stencil_k;
          int t_r = 0;
          int t_g = 0;
          int t_b = 0;

          for (stencil_j = -size; stencil_j <= size; stencil_j++) {
            for (stencil_k = -size; stencil_k <= size; stencil_k++) {
              t_r += p[i][CONV(j + stencil_j, k + stencil_k, width)].r;
              t_g += p[i][CONV(j + stencil_j, k + stencil_k, width)].g;
              t_b += p[i][CONV(j + stencil_j, k + stencil_k, width)].b;
            }
          }

          new[CONV(j, k, width)].r = t_r / ((2 * size + 1) * (2 * size + 1));
          new[CONV(j, k, width)].g = t_g / ((2 * size + 1) * (2 * size + 1));
          new[CONV(j, k, width)].b = t_b / ((2 * size + 1) * (2 * size + 1));
        }
      }

      for (j = 1; j < height - 1; j++) {
        for (k = 1; k < width - 1; k++) {

          float diff_r;
          float diff_g;
          float diff_b;

          diff_r = (new[CONV(j, k, width)].r - p[i][CONV(j, k, width)].r);
          diff_g = (new[CONV(j, k, width)].g - p[i][CONV(j, k, width)].g);
          diff_b = (new[CONV(j, k, width)].b - p[i][CONV(j, k, width)].b);

          if (diff_r > threshold || -diff_r > threshold || diff_g > threshold ||
              -diff_g > threshold || diff_b > threshold ||
              -diff_b > threshold) {
            end = 0;
          }

          p[i][CONV(j, k, width)].r = new[CONV(j, k, width)].r;
          p[i][CONV(j, k, width)].g = new[CONV(j, k, width)].g;
          p[i][CONV(j, k, width)].b = new[CONV(j, k, width)].b;
        }
      }

    } while (threshold > 0 && !end);

#if SOBELF_DEBUG
    printf("BLUR: number of iterations for image %d\n", n_iter);
#endif

    free(new);
  }
}
