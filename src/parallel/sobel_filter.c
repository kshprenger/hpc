#include "gif_math.h"
#include "gif_model.h"
#include <math.h>
#include <stdlib.h>

void apply_sobel_filter(animated_gif *image) {
  int i, j, k;
  int width, height;

  pixel **p;

  p = image->p;

  for (i = 0; i < image->n_images; i++) {
    width = image->width[i];
    height = image->height[i];

    pixel *sobel;

    sobel = (pixel *)malloc(width * height * sizeof(pixel));

    for (j = 1; j < height - 1; j++) {
      for (k = 1; k < width - 1; k++) {
        int pixel_blue_no, pixel_blue_n, pixel_blue_ne;
        int pixel_blue_so, pixel_blue_s, pixel_blue_se;
        int pixel_blue_o, pixel_blue, pixel_blue_e;

        float deltaX_blue;
        float deltaY_blue;
        float val_blue;

        pixel_blue_no = p[i][CONV(j - 1, k - 1, width)].b;
        pixel_blue_n = p[i][CONV(j - 1, k, width)].b;
        pixel_blue_ne = p[i][CONV(j - 1, k + 1, width)].b;
        pixel_blue_so = p[i][CONV(j + 1, k - 1, width)].b;
        pixel_blue_s = p[i][CONV(j + 1, k, width)].b;
        pixel_blue_se = p[i][CONV(j + 1, k + 1, width)].b;
        pixel_blue_o = p[i][CONV(j, k - 1, width)].b;
        pixel_blue = p[i][CONV(j, k, width)].b;
        pixel_blue_e = p[i][CONV(j, k + 1, width)].b;

        deltaX_blue = -pixel_blue_no + pixel_blue_ne - 2 * pixel_blue_o +
                      2 * pixel_blue_e - pixel_blue_so + pixel_blue_se;

        deltaY_blue = pixel_blue_se + 2 * pixel_blue_s + pixel_blue_so -
                      pixel_blue_ne - 2 * pixel_blue_n - pixel_blue_no;

        val_blue =
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

    for (j = 1; j < height - 1; j++) {
      for (k = 1; k < width - 1; k++) {
        p[i][CONV(j, k, width)].r = sobel[CONV(j, k, width)].r;
        p[i][CONV(j, k, width)].g = sobel[CONV(j, k, width)].g;
        p[i][CONV(j, k, width)].b = sobel[CONV(j, k, width)].b;
      }
    }

    free(sobel);
  }
}
