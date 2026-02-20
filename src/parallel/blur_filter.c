#include "gif_math.h"
#include "gif_model.h"
#include <omp.h>
#include <stdlib.h>

void apply_blur_filter(animated_gif *image, int size, int threshold) {

  if (image->n_images == 0)
    return;

  pixel **p = image->p;

  int width  = image->width[0];
  int height = image->height[0];

  int denom = (2 * size + 1) * (2 * size + 1);

  for (int i = 0; i < image->n_images; i++) {

    pixel *newp = (pixel *)malloc(width * height * sizeof(pixel));
    if (!newp) abort();

    int end;
    int n_iter = 0;

    do {
      end = 1;
      n_iter++;

      #pragma omp parallel for schedule(static)
      for (int idx = 0; idx < width * height; idx++) {
        newp[idx] = p[i][idx];
      }

      #pragma omp parallel for collapse(2) schedule(static)
      for (int j = size; j < height / 10 - size; j++) {
        for (int k = size; k < width - size; k++) {

          int t_r = 0, t_g = 0, t_b = 0;

          for (int sj = -size; sj <= size; sj++) {
            for (int sk = -size; sk <= size; sk++) {
              pixel q = p[i][CONV(j + sj, k + sk, width)];
              t_r += q.r;
              t_g += q.g;
              t_b += q.b;
            }
          }

          int idx = CONV(j, k, width);
          newp[idx].r = t_r / denom;
          newp[idx].g = t_g / denom;
          newp[idx].b = t_b / denom;
        }
      }

      #pragma omp parallel for collapse(2) schedule(static)
      for (int j = (height * 0.9) + size; j < height - size; j++) {
        for (int k = size; k < width - size; k++) {

          int t_r = 0, t_g = 0, t_b = 0;

          for (int sj = -size; sj <= size; sj++) {
            for (int sk = -size; sk <= size; sk++) {
              pixel q = p[i][CONV(j + sj, k + sk, width)];
              t_r += q.r;
              t_g += q.g;
              t_b += q.b;
            }
          }

          int idx = CONV(j, k, width);
          newp[idx].r = t_r / denom;
          newp[idx].g = t_g / denom;
          newp[idx].b = t_b / denom;
        }
      }

      #pragma omp parallel for collapse(2) reduction(&&:end) schedule(static)
      for (int j = 1; j < height - 1; j++) {
        for (int k = 1; k < width - 1; k++) {

          int idx = CONV(j, k, width);

          float dr = (float)newp[idx].r - (float)p[i][idx].r;
          float dg = (float)newp[idx].g - (float)p[i][idx].g;
          float db = (float)newp[idx].b - (float)p[i][idx].b;

          int ok = !(dr > threshold || -dr > threshold ||
                     dg > threshold || -dg > threshold ||
                     db > threshold || -db > threshold);

          end = end && ok;
        }
      }

      #pragma omp parallel for schedule(static)
      for (int idx = 0; idx < width * height; idx++) {
        p[i][idx] = newp[idx];
      }

    } while (threshold > 0 && !end);

#if SOBELF_DEBUG
    printf("BLUR: number of iterations for image %d\n", n_iter);
#endif

    free(newp);
  }
}
