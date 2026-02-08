#include "gif_lib.h"
#include "gif_model.h"
#include <stdio.h>
#include <stdlib.h>

animated_gif *load_pixels(char *filename) {
  GifFileType *g;
  ColorMapObject *colmap;
  int error;
  int n_images;
  int *width;
  int *height;
  pixel **p;
  int i;
  animated_gif *image;

  /* Open the GIF image (read mode) */
  g = DGifOpenFileName(filename, &error);
  if (g == NULL) {
    fprintf(stderr, "Error DGifOpenFileName %s\n", filename);
    return NULL;
  }

  /* Read the GIF image */
  error = DGifSlurp(g);
  if (error != GIF_OK) {
    fprintf(stderr, "Error DGifSlurp: %d <%s>\n", error,
            GifErrorString(g->Error));
    return NULL;
  }

  /* Grab the number of images and the size of each image */
  n_images = g->ImageCount;

  width = (int *)malloc(n_images * sizeof(int));
  if (width == NULL) {
    fprintf(stderr, "Unable to allocate width of size %d\n", n_images);
    return 0;
  }

  height = (int *)malloc(n_images * sizeof(int));
  if (height == NULL) {
    fprintf(stderr, "Unable to allocate height of size %d\n", n_images);
    return 0;
  }

  /* Fill the width and height */
  for (i = 0; i < n_images; i++) {
    width[i] = g->SavedImages[i].ImageDesc.Width;
    height[i] = g->SavedImages[i].ImageDesc.Height;

#if SOBELF_DEBUG
    printf("Image %d: l:%d t:%d w:%d h:%d interlace:%d localCM:%p\n", i,
           g->SavedImages[i].ImageDesc.Left, g->SavedImages[i].ImageDesc.Top,
           g->SavedImages[i].ImageDesc.Width,
           g->SavedImages[i].ImageDesc.Height,
           g->SavedImages[i].ImageDesc.Interlace,
           g->SavedImages[i].ImageDesc.ColorMap);
#endif
  }

  /* Get the global colormap */
  colmap = g->SColorMap;
  if (colmap == NULL) {
    fprintf(stderr, "Error global colormap is NULL\n");
    return NULL;
  }

#if SOBELF_DEBUG
  printf("Global color map: count:%d bpp:%d sort:%d\n",
         g->SColorMap->ColorCount, g->SColorMap->BitsPerPixel,
         g->SColorMap->SortFlag);
#endif

  /* Allocate the array of pixels to be returned */
  p = (pixel **)malloc(n_images * sizeof(pixel *));
  if (p == NULL) {
    fprintf(stderr, "Unable to allocate array of %d images\n", n_images);
    return NULL;
  }

  for (i = 0; i < n_images; i++) {
    p[i] = (pixel *)malloc(width[i] * height[i] * sizeof(pixel));
    if (p[i] == NULL) {
      fprintf(stderr, "Unable to allocate %d-th array of %d pixels\n", i,
              width[i] * height[i]);
      return NULL;
    }
  }

  /* Fill pixels */

  /* For each image */
  for (i = 0; i < n_images; i++) {
    int j;

    /* Get the local colormap if needed */
    if (g->SavedImages[i].ImageDesc.ColorMap) {

      /* TODO No support for local color map */
      fprintf(stderr, "Error: application does not support local colormap\n");
      return NULL;

      colmap = g->SavedImages[i].ImageDesc.ColorMap;
    }

    /* Traverse the image and fill pixels */
    for (j = 0; j < width[i] * height[i]; j++) {
      int c;

      c = g->SavedImages[i].RasterBits[j];

      p[i][j].r = colmap->Colors[c].Red;
      p[i][j].g = colmap->Colors[c].Green;
      p[i][j].b = colmap->Colors[c].Blue;
    }
  }

  /* Allocate image info */
  image = (animated_gif *)malloc(sizeof(animated_gif));
  if (image == NULL) {
    fprintf(stderr, "Unable to allocate memory for animated_gif\n");
    return NULL;
  }

  /* Fill image fields */
  image->n_images = n_images;
  image->width = width;
  image->height = height;
  image->p = p;
  image->g = g;

#if SOBELF_DEBUG
  printf("-> GIF w/ %d image(s) with first image of size %d x %d\n",
         image->n_images, image->width[0], image->height[0]);
#endif

  return image;
}
