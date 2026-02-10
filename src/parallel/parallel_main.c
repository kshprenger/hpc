#include "filter_api.h"
#include "gif_lib.h"
#include "gif_model.h"
#include "persist_api.h"
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

int main(int argc, char **argv) {
  char *input_filename;
  char *output_filename;
  animated_gif *image = NULL;
  animated_gif *local_image = NULL;
  struct timeval t1, t2;
  double duration, max_duration;
  int rank, size;
  int local_n_images, start_frame, end_frame;

  /* Initialize MPI */
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  /* Check command-line arguments */
  if (argc < 3) {
    if (rank == 0) {
      fprintf(stderr, "Usage: %s input.gif output.gif \n", argv[0]);
    }
    MPI_Finalize();
    return 1;
  }

  input_filename = argv[1];
  output_filename = argv[2];

  /* If only one MPI rank, run sequentially without MPI overhead */
  if (size == 1) {
    /* IMPORT Timer start */
    gettimeofday(&t1, NULL);

    image = load_pixels(input_filename);
    if (image == NULL) {
      MPI_Finalize();
      return 1;
    }

    /* IMPORT Timer stop */
    gettimeofday(&t2, NULL);
    duration = (t2.tv_sec - t1.tv_sec) + ((t2.tv_usec - t1.tv_usec) / 1e6);
    printf("GIF loaded from file %s with %d image(s) in %lf s\n",
           input_filename, image->n_images, duration);

    /* FILTER Timer start */
    gettimeofday(&t1, NULL);

    apply_gray_filter(image);

    apply_blur_filter(image, 5, 20);

    apply_sobel_filter(image);

    /* FILTER Timer stop */
    gettimeofday(&t2, NULL);
    duration = (t2.tv_sec - t1.tv_sec) + ((t2.tv_usec - t1.tv_usec) / 1e6);
    printf("SOBEL done in %lf s\n", duration);

    /* EXPORT Timer start */
    gettimeofday(&t1, NULL);

    if (!store_pixels(output_filename, image)) {
      MPI_Finalize();
      return 1;
    }

    /* EXPORT Timer stop */
    gettimeofday(&t2, NULL);
    duration = (t2.tv_sec - t1.tv_sec) + ((t2.tv_usec - t1.tv_usec) / 1e6);
    printf("Export done in %lf s in file %s\n", duration, output_filename);

    MPI_Finalize();
    return 0;
  }

  /* IMPORT Timer start */
  if (rank == 0) {
    gettimeofday(&t1, NULL);
  }

  if (rank == 0) {
    image = load_pixels(input_filename);
    if (image == NULL) {
      MPI_Abort(MPI_COMM_WORLD, 1);
      return 1;
    }
  }

  /* Broadcast the number of images */
  int total_images = 0;
  if (rank == 0) {
    total_images = image->n_images;
    gettimeofday(&t2, NULL);
    duration = (t2.tv_sec - t1.tv_sec) + ((t2.tv_usec - t1.tv_usec) / 1e6);
    printf("GIF loaded from file %s with %d image(s) in %lf s\n",
           input_filename, total_images, duration);
  }
  MPI_Bcast(&total_images, 1, MPI_INT, 0, MPI_COMM_WORLD);

  /* Calculate work distribution */
  int images_per_rank = total_images / size;
  int remainder = total_images % size;

  if (rank < remainder) {
    local_n_images = images_per_rank + 1;
    start_frame = rank * local_n_images;
  } else {
    local_n_images = images_per_rank;
    start_frame = remainder * (images_per_rank + 1) +
                  (rank - remainder) * images_per_rank;
  }
  end_frame = start_frame + local_n_images;

  if (rank == 0) {
    printf("Distributing %d frames across %d MPI ranks\n", total_images, size);
  }

  local_image = (animated_gif *)malloc(sizeof(animated_gif));
  local_image->n_images = local_n_images;
  local_image->width = (int *)malloc(local_n_images * sizeof(int));
  local_image->height = (int *)malloc(local_n_images * sizeof(int));
  local_image->p = (pixel **)malloc(local_n_images * sizeof(pixel *));
  local_image->g = NULL;

  /* Distribute image metadata and pixels */
  if (rank == 0) {
    /* Rank 0: Send metadata to all other ranks */
    for (int dest = 1; dest < size; dest++) {
      int dest_start, dest_n_images;
      if (dest < remainder) {
        dest_n_images = images_per_rank + 1;
        dest_start = dest * dest_n_images;
      } else {
        dest_n_images = images_per_rank;
        dest_start = remainder * (images_per_rank + 1) +
                     (dest - remainder) * images_per_rank;
      }

      for (int j = 0; j < dest_n_images; j++) {
        int frame = dest_start + j;
        if (frame < total_images) {
          int metadata[2] = {image->width[frame], image->height[frame]};
          MPI_Send(metadata, 2, MPI_INT, dest, frame, MPI_COMM_WORLD);
        }
      }
    }

    /* Rank 0: Copy own metadata */
    for (int i = 0; i < local_n_images; i++) {
      int global_frame = start_frame + i;
      local_image->width[i] = image->width[global_frame];
      local_image->height[i] = image->height[global_frame];
      local_image->p[i] =
          (pixel *)malloc(image->width[global_frame] *
                          image->height[global_frame] * sizeof(pixel));
    }

    /* Rank 0: Send pixel data to all other ranks */
    for (int dest = 1; dest < size; dest++) {
      int dest_start, dest_n_images;
      if (dest < remainder) {
        dest_n_images = images_per_rank + 1;
        dest_start = dest * dest_n_images;
      } else {
        dest_n_images = images_per_rank;
        dest_start = remainder * (images_per_rank + 1) +
                     (dest - remainder) * images_per_rank;
      }

      for (int j = 0; j < dest_n_images; j++) {
        int frame = dest_start + j;
        if (frame < total_images) {
          int size_pixels =
              image->width[frame] * image->height[frame] * sizeof(pixel);
          MPI_Send(image->p[frame], size_pixels, MPI_BYTE, dest, frame,
                   MPI_COMM_WORLD);
        }
      }
    }

    /* Rank 0: Copy own pixel data */
    for (int i = 0; i < local_n_images; i++) {
      int global_frame = start_frame + i;
      int size_pixels = image->width[global_frame] *
                        image->height[global_frame] * sizeof(pixel);
      memcpy(local_image->p[i], image->p[global_frame], size_pixels);
    }

  } else {
    /* Other ranks: Receive metadata */
    for (int i = 0; i < local_n_images; i++) {
      int global_frame = start_frame + i;
      int metadata[2];
      MPI_Recv(metadata, 2, MPI_INT, 0, global_frame, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
      local_image->width[i] = metadata[0];
      local_image->height[i] = metadata[1];
      local_image->p[i] =
          (pixel *)malloc(metadata[0] * metadata[1] * sizeof(pixel));
    }

    /* Other ranks: Receive pixel data */
    for (int i = 0; i < local_n_images; i++) {
      int global_frame = start_frame + i;
      int size_pixels =
          local_image->width[i] * local_image->height[i] * sizeof(pixel);
      MPI_Recv(local_image->p[i], size_pixels, MPI_BYTE, 0, global_frame,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);

  /* FILTER Timer start */
  gettimeofday(&t1, NULL);

  if (local_n_images > 0) {
    apply_gray_filter(local_image);
    apply_blur_filter(local_image, 5, 20);
    apply_sobel_filter(local_image);
  }

  /* FILTER Timer stop */
  gettimeofday(&t2, NULL);
  duration = (t2.tv_sec - t1.tv_sec) + ((t2.tv_usec - t1.tv_usec) / 1e6);

  /* Find maximum duration across all ranks */
  MPI_Reduce(&duration, &max_duration, 1, MPI_DOUBLE, MPI_MAX, 0,
             MPI_COMM_WORLD);

  if (rank == 0) {
    printf("SOBEL done in %lf s (max across all ranks)\n", max_duration);
  }

  /* Gather results back to rank 0 */
  if (rank == 0) {
    /* Copy own results */
    for (int i = 0; i < local_n_images; i++) {
      int global_frame = start_frame + i;
      memcpy(image->p[global_frame], local_image->p[i],
             image->width[global_frame] * image->height[global_frame] *
                 sizeof(pixel));
    }

    /* Receive from other ranks */
    for (int src = 1; src < size; src++) {
      int src_start, src_n_images;
      if (src < remainder) {
        src_n_images = images_per_rank + 1;
        src_start = src * src_n_images;
      } else {
        src_n_images = images_per_rank;
        src_start = remainder * (images_per_rank + 1) +
                    (src - remainder) * images_per_rank;
      }

      for (int j = 0; j < src_n_images; j++) {
        int frame = src_start + j;
        if (frame < total_images) {
          int size_pixels =
              image->width[frame] * image->height[frame] * sizeof(pixel);
          MPI_Recv(image->p[frame], size_pixels, MPI_BYTE, src, frame,
                   MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
      }
    }
  } else {
    /* Send results to rank 0 */
    for (int i = 0; i < local_n_images; i++) {
      int global_frame = start_frame + i;
      int size_pixels =
          local_image->width[i] * local_image->height[i] * sizeof(pixel);
      MPI_Send(local_image->p[i], size_pixels, MPI_BYTE, 0, global_frame,
               MPI_COMM_WORLD);
    }
  }

  /* EXPORT Timer start (only rank 0) */
  if (rank == 0) {
    gettimeofday(&t1, NULL);

    /* Store file from array of pixels to GIF file */
    if (!store_pixels(output_filename, image)) {
      MPI_Abort(MPI_COMM_WORLD, 1);
      return 1;
    }

    /* EXPORT Timer stop */
    gettimeofday(&t2, NULL);
    duration = (t2.tv_sec - t1.tv_sec) + ((t2.tv_usec - t1.tv_usec) / 1e6);
    printf("Export done in %lf s in file %s\n", duration, output_filename);
  }

  if (local_image) {
    for (int i = 0; i < local_n_images; i++) {
      free(local_image->p[i]);
    }
    free(local_image->p);
    free(local_image->width);
    free(local_image->height);
    free(local_image);
  }

  MPI_Finalize();
  return 0;
}
