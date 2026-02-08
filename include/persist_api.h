#ifndef _PERSIST_API_H_
#define _PERSIST_API_H_
#include "gif_model.h"
animated_gif *load_pixels(char *filename);
int store_pixels(char *filename, animated_gif *image);
#endif
