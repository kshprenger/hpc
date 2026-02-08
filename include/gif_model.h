#ifndef _GIF_MODEL_H_
#define _GIF_MODEL_H_

#include "gif_lib.h"

/* Represent one pixel from the image */
typedef struct pixel {
  int r; /* Red */
  int g; /* Green */
  int b; /* Blue */
} pixel;

/* Represent one GIF image (animated or not */
typedef struct animated_gif {
  int n_images;   /* Number of images */
  int *width;     /* Width of each image */
  int *height;    /* Height of each image */
  pixel **p;      /* Pixels of each image */
  GifFileType *g; /* Internal representation.
                     DO NOT MODIFY */
} animated_gif;

#endif
