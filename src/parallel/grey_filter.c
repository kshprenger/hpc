#include "gif_model.h"
#include <omp.h>

void apply_gray_filter(animated_gif *image) {
  if (image->n_images == 0) {
    return;
  }
  
  pixel **p = image->p;
  int n = image->width[0] * image->height[0];
  for (int i = 0; i < image->n_images; i++) {
    #pragma omp parallel for schedule(static)
    for (int j = 0; j < n; j++) {
      int moy = (p[i][j].r + p[i][j].g + p[i][j].b) / 3;

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
