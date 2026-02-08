#include "gif_model.h"

void apply_gray_filter(animated_gif *image) {
  int i, j;
  pixel **p;

  p = image->p;

  for (i = 0; i < image->n_images; i++) {
    for (j = 0; j < image->width[i] * image->height[i]; j++) {
      int moy;

      moy = (p[i][j].r + p[i][j].g + p[i][j].b) / 3;
      if (moy < 0)
        moy = 0;
      if (moy > 255)
        moy = 255;

      p[i][j].r = moy;
      p[i][j].g = moy;
      p[i][j].b = moy;
    }
  }
}
