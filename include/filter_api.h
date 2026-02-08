#ifndef _FILTER_API_H_
#define _FILTER_API_H_
#include "gif_model.h"
void apply_blur_filter(animated_gif *image, int size, int threshold);
void apply_gray_filter(animated_gif *image);
void apply_sobel_filter(animated_gif *image);
#endif
