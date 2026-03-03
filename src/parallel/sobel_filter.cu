#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern "C" {
#include "gif_math.h"
#include "gif_model.h"
}

#ifndef GHOST_WIDTH
#define GHOST_WIDTH 5
#endif

#define MIN_PIXELS_FOR_GPU (128 * 128)

#define BLOCK_SIZE 16
#define BLUR_MARGIN 5
#define SECTION_WIDTH (BLOCK_SIZE + 2 * BLUR_MARGIN)
#define SECTION_HEIGHT (BLOCK_SIZE + 2 * BLUR_MARGIN)

static pixel *host_input = NULL;
static pixel *host_output = NULL;
static pixel *device_current = NULL;
static pixel *device_next = NULL;
static int *device_converged_flag = NULL;
static int *host_converged_flag = NULL;
static int buffer_capacity = 0;

static int grow_buffers(int num_pixels) {
  if (num_pixels <= buffer_capacity)
    return 1;

  if (host_input) {
    cudaFreeHost(host_input);
    host_input = NULL;
  }
  if (host_output) {
    cudaFreeHost(host_output);
    host_output = NULL;
  }
  if (device_current) {
    cudaFree(device_current);
    device_current = NULL;
  }
  if (device_next) {
    cudaFree(device_next);
    device_next = NULL;
  }
  if (device_converged_flag) {
    cudaFree(device_converged_flag);
    device_converged_flag = NULL;
  }
  if (host_converged_flag) {
    cudaFreeHost(host_converged_flag);
    host_converged_flag = NULL;
  }
  buffer_capacity = 0;

  size_t bytes = (size_t)num_pixels * sizeof(pixel);
  if (cudaMallocHost(&host_input, bytes) != cudaSuccess)
    return 0;
  if (cudaMallocHost(&host_output, bytes) != cudaSuccess)
    return 0;
  if (cudaMalloc(&device_current, bytes) != cudaSuccess)
    return 0;
  if (cudaMalloc(&device_next, bytes) != cudaSuccess)
    return 0;
  if (cudaMalloc(&device_converged_flag, sizeof(int)) != cudaSuccess)
    return 0;
  if (cudaMallocHost(&host_converged_flag, sizeof(int)) != cudaSuccess)
    return 0;

  buffer_capacity = num_pixels;
  return 1;
}

__global__ void grayscale_kernel(pixel *pixels, int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height)
    return;
  int index = y * width + x;
  int value = (pixels[index].r + pixels[index].g + pixels[index].b) / 3;
  value = value < 0 ? 0 : (value > 255 ? 255 : value);
  pixels[index].r = pixels[index].g = pixels[index].b = value;
}

__global__ void blur_kernel(const pixel *__restrict__ source,
                            pixel *__restrict__ dest, int width, int height,
                            int row_start, int row_end, int col_start,
                            int col_end, int radius, int denominator) {
  __shared__ pixel section[SECTION_HEIGHT][SECTION_WIDTH];

  int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
  int section_col_start = (int)(blockIdx.x * BLOCK_SIZE) - radius;
  int section_row_start = (int)(blockIdx.y * BLOCK_SIZE) - radius;
  int thread_id = threadIdx.y * BLOCK_SIZE + threadIdx.x;
  int block_threads = BLOCK_SIZE * BLOCK_SIZE;
  int section_pixels = SECTION_HEIGHT * SECTION_WIDTH;

  for (int i = thread_id; i < section_pixels; i += block_threads) {
    int section_col = i % SECTION_WIDTH, section_row = i / SECTION_WIDTH;
    int global_col = section_col_start + section_col,
        global_row = section_row_start + section_row;
    if (global_col >= 0 && global_col < width && global_row >= 0 &&
        global_row < height)
      section[section_row][section_col] =
          source[global_row * width + global_col];
    else
      section[section_row][section_col] = {0, 0, 0};
  }
  __syncthreads();

  if (row < row_start || row >= row_end || col < col_start || col >= col_end ||
      row >= height || col >= width)
    return;

  int local_row = threadIdx.y + radius, local_col = threadIdx.x + radius;
  int total_red = 0, total_green = 0, total_blue = 0;
  for (int dy = -radius; dy <= radius; dy++)
    for (int dx = -radius; dx <= radius; dx++) {
      pixel neighbor = section[local_row + dy][local_col + dx];
      total_red += neighbor.r;
      total_green += neighbor.g;
      total_blue += neighbor.b;
    }
  int index = row * width + col;
  dest[index].r = total_red / denominator;
  dest[index].g = total_green / denominator;
  dest[index].b = total_blue / denominator;
}

__global__ void convergence_kernel(const pixel *__restrict__ current,
                                   const pixel *__restrict__ next, int width,
                                   int height, int col_start, int col_end,
                                   int threshold, int *flag) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row < 1 || row >= height - 1 || col < col_start || col >= col_end)
    return;
  int index = row * width + col;
  int diff_red = abs(next[index].r - current[index].r);
  int diff_green = abs(next[index].g - current[index].g);
  int diff_blue = abs(next[index].b - current[index].b);
  if (diff_red > threshold || diff_green > threshold || diff_blue > threshold)
    atomicAnd(flag, 0);
}

__global__ void sobel_kernel(const pixel *__restrict__ source,
                             pixel *__restrict__ dest, int width, int height) {
  extern __shared__ pixel shared_section[];
  int tx = threadIdx.x, ty = threadIdx.y;
  int x = blockIdx.x * blockDim.x + tx;
  int y = blockIdx.y * blockDim.y + ty;
  int section_width = blockDim.x + 2;
  int sx = tx + 1, sy = ty + 1;

  if (x < width && y < height)
    shared_section[sy * section_width + sx] = source[y * width + x];
  if (tx == 0 && x > 0)
    shared_section[sy * section_width + 0] = source[y * width + (x - 1)];
  if (tx == blockDim.x - 1 && x < width - 1)
    shared_section[sy * section_width + sx + 1] = source[y * width + (x + 1)];
  if (ty == 0 && y > 0)
    shared_section[0 * section_width + sx] = source[(y - 1) * width + x];
  if (ty == blockDim.y - 1 && y < height - 1)
    shared_section[(sy + 1) * section_width + sx] = source[(y + 1) * width + x];
  if (tx == 0 && ty == 0 && x > 0 && y > 0)
    shared_section[0 * section_width + 0] = source[(y - 1) * width + (x - 1)];
  if (tx == blockDim.x - 1 && ty == 0 && x < width - 1 && y > 0)
    shared_section[0 * section_width + sx + 1] =
        source[(y - 1) * width + (x + 1)];
  if (tx == 0 && ty == blockDim.y - 1 && x > 0 && y < height - 1)
    shared_section[(sy + 1) * section_width + 0] =
        source[(y + 1) * width + (x - 1)];
  if (tx == blockDim.x - 1 && ty == blockDim.y - 1 && x < width - 1 &&
      y < height - 1)
    shared_section[(sy + 1) * section_width + sx + 1] =
        source[(y + 1) * width + (x + 1)];
  __syncthreads();

  if (x <= 0 || x >= width - 1 || y <= 0 || y >= height - 1)
    return;

  int gradient_x = -shared_section[(sy - 1) * section_width + (sx - 1)].b -
                   2 * shared_section[sy * section_width + (sx - 1)].b -
                   shared_section[(sy + 1) * section_width + (sx - 1)].b +
                   shared_section[(sy - 1) * section_width + (sx + 1)].b +
                   2 * shared_section[sy * section_width + (sx + 1)].b +
                   shared_section[(sy + 1) * section_width + (sx + 1)].b;
  int gradient_y = shared_section[(sy + 1) * section_width + (sx - 1)].b +
                   2 * shared_section[(sy + 1) * section_width + sx].b +
                   shared_section[(sy + 1) * section_width + (sx + 1)].b -
                   shared_section[(sy - 1) * section_width + (sx - 1)].b -
                   2 * shared_section[(sy - 1) * section_width + sx].b -
                   shared_section[(sy - 1) * section_width + (sx + 1)].b;

  float magnitude =
      sqrtf((float)(gradient_x * gradient_x + gradient_y * gradient_y)) / 4.0f;
  int index = y * width + x;
  int value = (magnitude > 50.0f) ? 255 : 0;
  dest[index].r = dest[index].g = dest[index].b = value;
}

extern "C" int cuda_is_available(void) {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0)
    return 0;
  for (int i = 0; i < device_count; i++) {
    cudaDeviceProp properties;
    if (cudaGetDeviceProperties(&properties, i) == cudaSuccess &&
        properties.major >= 2)
      return 1;
  }
  return 0;
}

extern "C" void cuda_get_device_name(char *name, int maxlen) {
  int device;
  if (cudaGetDevice(&device) != cudaSuccess) {
    snprintf(name, maxlen, "Unknown");
    return;
  }
  cudaDeviceProp properties;
  if (cudaGetDeviceProperties(&properties, device) == cudaSuccess)
    snprintf(name, maxlen, "%s", properties.name);
  else
    snprintf(name, maxlen, "Unknown");
}

extern "C" void apply_gray_filter_to_region_cuda(pixel *pixels, int width,
                                                 int height) {
  if (!pixels || width <= 0 || height <= 0)
    return;
  int num_pixels = width * height;

  if (num_pixels < MIN_PIXELS_FOR_GPU) {
    for (int i = 0; i < num_pixels; i++) {
      int value = (pixels[i].r + pixels[i].g + pixels[i].b) / 3;
      value = value < 0 ? 0 : (value > 255 ? 255 : value);
      pixels[i].r = pixels[i].g = pixels[i].b = value;
    }
    return;
  }

  if (!grow_buffers(num_pixels))
    return;

  size_t bytes = (size_t)num_pixels * sizeof(pixel);
  memcpy(host_input, pixels, bytes);
  cudaMemcpy(device_current, host_input, bytes, cudaMemcpyHostToDevice);

  dim3 block(16, 16), grid((width + 15) / 16, (height + 15) / 16);
  grayscale_kernel<<<grid, block>>>(device_current, width, height);
  cudaDeviceSynchronize();

  cudaMemcpy(host_input, device_current, bytes, cudaMemcpyDeviceToHost);
  memcpy(pixels, host_input, bytes);
}

extern "C" void apply_blur_filter_to_region_cuda(pixel *pixels, int width,
                                                 int height, int radius,
                                                 int threshold, int region_id,
                                                 int num_regions) {
  if (!pixels || width <= 0 || height <= 0)
    return;
  int num_pixels = width * height;

  if (num_pixels < MIN_PIXELS_FOR_GPU)
    return;

  if (!grow_buffers(num_pixels))
    return;

  size_t bytes = (size_t)num_pixels * sizeof(pixel);
  memcpy(host_input, pixels, bytes);
  cudaMemcpy(device_current, host_input, bytes, cudaMemcpyHostToDevice);

  int col_start = (region_id == 0) ? 1 : GHOST_WIDTH;
  int col_end =
      (region_id == num_regions - 1) ? width - 1 : width - GHOST_WIDTH;
  int blur_col_start = (col_start > radius) ? col_start : radius;
  int blur_col_end = (col_end < width - radius) ? col_end : width - radius;

  int top_row_start = radius, top_row_end = height / 10 - radius;
  int bottom_row_start = (int)(height * 0.9f) + radius,
      bottom_row_end = height - radius;

  int denominator = (2 * radius + 1) * (2 * radius + 1);

  dim3 blur_block(BLOCK_SIZE, BLOCK_SIZE),
      blur_grid((width + BLOCK_SIZE - 1) / BLOCK_SIZE,
                (height + BLOCK_SIZE - 1) / BLOCK_SIZE);
  dim3 conv_block(16, 16), conv_grid((width + 15) / 16, (height + 15) / 16);

  pixel *current = device_current, *next = device_next;
  int converged = 0;

  do {
    cudaMemcpy(next, current, bytes, cudaMemcpyDeviceToDevice);

    if (top_row_start < top_row_end)
      blur_kernel<<<blur_grid, blur_block>>>(
          current, next, width, height, top_row_start, top_row_end,
          blur_col_start, blur_col_end, radius, denominator);
    if (bottom_row_start < bottom_row_end)
      blur_kernel<<<blur_grid, blur_block>>>(
          current, next, width, height, bottom_row_start, bottom_row_end,
          blur_col_start, blur_col_end, radius, denominator);

    if (threshold > 0) {
      *host_converged_flag = 1;
      cudaMemcpy(device_converged_flag, host_converged_flag, sizeof(int),
                 cudaMemcpyHostToDevice);
      convergence_kernel<<<conv_grid, conv_block>>>(
          current, next, width, height, col_start, col_end, threshold,
          device_converged_flag);
      cudaDeviceSynchronize();
      cudaMemcpy(host_converged_flag, device_converged_flag, sizeof(int),
                 cudaMemcpyDeviceToHost);
      converged = *host_converged_flag;
    } else {
      cudaDeviceSynchronize();
      converged = 1;
    }

    pixel *temp = current;
    current = next;
    next = temp;
  } while (!converged);

  cudaMemcpy(host_input, current, bytes, cudaMemcpyDeviceToHost);
  memcpy(pixels, host_input, bytes);
}

extern "C" void apply_sobel_filter_to_region_cuda(pixel *pixels, int width,
                                                  int height, int ghost_width,
                                                  int region_id,
                                                  int num_regions) {
  if (!pixels || width <= 0 || height <= 0)
    return;
  int num_pixels = width * height;

  if (num_pixels < MIN_PIXELS_FOR_GPU) {
    pixel *temp = (pixel *)malloc(num_pixels * sizeof(pixel));
    if (!temp)
      return;
    for (int j = 1; j < height - 1; j++)
      for (int k = 1; k < width - 1; k++) {
        int northwest = pixels[CONV(j - 1, k - 1, width)].b,
            north = pixels[CONV(j - 1, k, width)].b,
            northeast = pixels[CONV(j - 1, k + 1, width)].b;
        int southwest = pixels[CONV(j + 1, k - 1, width)].b,
            south = pixels[CONV(j + 1, k, width)].b,
            southeast = pixels[CONV(j + 1, k + 1, width)].b;
        int west = pixels[CONV(j, k - 1, width)].b,
            east = pixels[CONV(j, k + 1, width)].b;
        float dx = -northwest + northeast - 2.0f * west + 2.0f * east -
                   southwest + southeast;
        float dy = southeast + 2.0f * south + southwest - northeast -
                   2.0f * north - northwest;
        int value = (sqrtf(dx * dx + dy * dy) / 4.0f) > 50 ? 255 : 0;
        temp[CONV(j, k, width)] = {value, value, value};
      }
    int col_start = (region_id == 0) ? 1 : ghost_width;
    int col_end =
        (region_id == num_regions - 1) ? width - 1 : width - ghost_width;
    for (int j = 1; j < height - 1; j++)
      for (int k = col_start; k < col_end; k++)
        pixels[CONV(j, k, width)] = temp[CONV(j, k, width)];
    free(temp);
    return;
  }

  if (!grow_buffers(num_pixels))
    return;

  size_t bytes = (size_t)num_pixels * sizeof(pixel);
  memcpy(host_input, pixels, bytes);
  cudaMemcpy(device_current, host_input, bytes, cudaMemcpyHostToDevice);

  dim3 block(16, 16), grid((width + 15) / 16, (height + 15) / 16);
  size_t shared_memory = (size_t)(16 + 2) * (16 + 2) * sizeof(pixel);
  sobel_kernel<<<grid, block, shared_memory>>>(device_current, device_next,
                                               width, height);
  cudaDeviceSynchronize();

  cudaMemcpy(host_output, device_next, bytes, cudaMemcpyDeviceToHost);

  int col_start = (region_id == 0) ? 1 : ghost_width;
  int col_end =
      (region_id == num_regions - 1) ? width - 1 : width - ghost_width;
  for (int j = 1; j < height - 1; j++)
    for (int k = col_start; k < col_end; k++)
      pixels[CONV(j, k, width)] = host_output[CONV(j, k, width)];
}

extern "C" void apply_gray_sobel_fused_to_region_cuda(pixel *pixels, int width,
                                                      int height,
                                                      int ghost_width,
                                                      int region_id,
                                                      int num_regions) {
  if (!pixels || width <= 0 || height <= 0)
    return;
  int num_pixels = width * height;

  if (num_pixels < MIN_PIXELS_FOR_GPU) {
    for (int i = 0; i < num_pixels; i++) {
      int value = (pixels[i].r + pixels[i].g + pixels[i].b) / 3;
      pixels[i].r = pixels[i].g = pixels[i].b = value;
    }
    pixel *temp = (pixel *)malloc(num_pixels * sizeof(pixel));
    if (!temp)
      return;
    for (int j = 1; j < height - 1; j++)
      for (int k = 1; k < width - 1; k++) {
        int northwest = pixels[CONV(j - 1, k - 1, width)].b,
            north = pixels[CONV(j - 1, k, width)].b,
            northeast = pixels[CONV(j - 1, k + 1, width)].b;
        int southwest = pixels[CONV(j + 1, k - 1, width)].b,
            south = pixels[CONV(j + 1, k, width)].b,
            southeast = pixels[CONV(j + 1, k + 1, width)].b;
        int west = pixels[CONV(j, k - 1, width)].b,
            east = pixels[CONV(j, k + 1, width)].b;
        float dx = -northwest + northeast - 2.0f * west + 2.0f * east -
                   southwest + southeast;
        float dy = southeast + 2.0f * south + southwest - northeast -
                   2.0f * north - northwest;
        int value = (sqrtf(dx * dx + dy * dy) / 4.0f) > 50 ? 255 : 0;
        temp[CONV(j, k, width)] = {value, value, value};
      }
    int col_start = (region_id == 0) ? 1 : ghost_width;
    int col_end =
        (region_id == num_regions - 1) ? width - 1 : width - ghost_width;
    for (int j = 1; j < height - 1; j++)
      for (int k = col_start; k < col_end; k++)
        pixels[CONV(j, k, width)] = temp[CONV(j, k, width)];
    free(temp);
    return;
  }

  if (!grow_buffers(num_pixels))
    return;

  size_t bytes = (size_t)num_pixels * sizeof(pixel);
  memcpy(host_input, pixels, bytes);
  cudaMemcpy(device_current, host_input, bytes, cudaMemcpyHostToDevice);

  dim3 block(16, 16), grid((width + 15) / 16, (height + 15) / 16);
  size_t shared_memory = (size_t)(16 + 2) * (16 + 2) * sizeof(pixel);

  grayscale_kernel<<<grid, block>>>(device_current, width, height);
  cudaDeviceSynchronize();
  sobel_kernel<<<grid, block, shared_memory>>>(device_current, device_next,
                                               width, height);
  cudaDeviceSynchronize();

  cudaMemcpy(host_output, device_next, bytes, cudaMemcpyDeviceToHost);

  int col_start = (region_id == 0) ? 1 : ghost_width;
  int col_end =
      (region_id == num_regions - 1) ? width - 1 : width - ghost_width;
  for (int j = 1; j < height - 1; j++)
    for (int k = col_start; k < col_end; k++)
      pixels[CONV(j, k, width)] = host_output[CONV(j, k, width)];
}

extern "C" void apply_sobel_filter_cuda(animated_gif *image) {
  if (!image || image->n_images == 0)
    return;
  int width = image->width[0], height = image->height[0],
      num_pixels = width * height;
  if (!grow_buffers(num_pixels))
    return;
  size_t bytes = (size_t)num_pixels * sizeof(pixel);
  dim3 block(16, 16), grid((width + 15) / 16, (height + 15) / 16);
  size_t shared_memory = (size_t)(16 + 2) * (16 + 2) * sizeof(pixel);
  for (int i = 0; i < image->n_images; i++) {
    memcpy(host_input, image->p[i], bytes);
    cudaMemcpy(device_current, host_input, bytes, cudaMemcpyHostToDevice);
    sobel_kernel<<<grid, block, shared_memory>>>(device_current, device_next,
                                                 width, height);
    cudaDeviceSynchronize();
    cudaMemcpy(host_output, device_next, bytes, cudaMemcpyDeviceToHost);
    for (int j = 1; j < height - 1; j++)
      for (int k = 1; k < width - 1; k++)
        image->p[i][CONV(j, k, width)] = host_output[CONV(j, k, width)];
  }
}
