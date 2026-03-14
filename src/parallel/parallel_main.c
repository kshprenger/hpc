#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include "runtime_config.h"


static void print_usage(const char *prog) {
  fprintf(stderr,
          "Usage: %s [--mpi off|auto|full|hybrid] "
          "[--openmp off|auto|force] "
          "[--cuda off|auto|force] "
          "input.gif output.gif\n",
          prog);
}

static int parse_args(int argc, char **argv,
                      runtime_config_t *cfg,
                      char **input_filename,
                      char **output_filename) {
  cfg->mpi_mode = MPI_MODE_AUTO;
  cfg->openmp_mode = OPENMP_MODE_AUTO;
  cfg->cuda_mode = CUDA_MODE_AUTO;

  int positional = 0;

  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "--mpi") == 0) {
      if (i + 1 >= argc) return 0;
      i++;
      if (strcmp(argv[i], "off") == 0) cfg->mpi_mode = MPI_MODE_OFF;
      else if (strcmp(argv[i], "auto") == 0) cfg->mpi_mode = MPI_MODE_AUTO;
      else if (strcmp(argv[i], "full") == 0) cfg->mpi_mode = MPI_MODE_FULL;
      else if (strcmp(argv[i], "hybrid") == 0) cfg->mpi_mode = MPI_MODE_HYBRID;
      else return 0;
    } else if (strcmp(argv[i], "--openmp") == 0) {
      if (i + 1 >= argc) return 0;
      i++;
      if (strcmp(argv[i], "off") == 0) cfg->openmp_mode = OPENMP_MODE_OFF;
      else if (strcmp(argv[i], "auto") == 0) cfg->openmp_mode = OPENMP_MODE_AUTO;
      else if (strcmp(argv[i], "force") == 0) cfg->openmp_mode = OPENMP_MODE_FORCE;
      else return 0;
    } else if (strcmp(argv[i], "--cuda") == 0) {
      if (i + 1 >= argc) return 0;
      i++;
      if (strcmp(argv[i], "off") == 0) cfg->cuda_mode = CUDA_MODE_OFF;
      else if (strcmp(argv[i], "auto") == 0) cfg->cuda_mode = CUDA_MODE_AUTO;
      else if (strcmp(argv[i], "force") == 0) cfg->cuda_mode = CUDA_MODE_FORCE;
      else return 0;
    } else if (argv[i][0] == '-') {
      return 0;
    } else {
      if (positional == 0) {
        *input_filename = argv[i];
      } else if (positional == 1) {
        *output_filename = argv[i];
      } else {
        return 0;
      }
      positional++;
    }
  }

  return (positional == 2);
}

static const char *mpi_mode_name(mpi_mode_t mode) {
  switch (mode) {
    case MPI_MODE_OFF:    return "off";
    case MPI_MODE_FULL:   return "full";
    case MPI_MODE_HYBRID: return "hybrid";
    case MPI_MODE_AUTO:
    default:              return "auto";
  }
}

static const char *openmp_mode_name(openmp_mode_t mode) {
  switch (mode) {
    case OPENMP_MODE_OFF:   return "off";
    case OPENMP_MODE_FORCE: return "force";
    case OPENMP_MODE_AUTO:
    default:                return "auto";
  }
}

static const char *cuda_mode_name(cuda_mode_t mode) {
  switch (mode) {
    case CUDA_MODE_OFF:   return "off";
    case CUDA_MODE_FORCE: return "force";
    case CUDA_MODE_AUTO:
    default:              return "auto";
  }
}

extern void Master(char *input_file, char *output_file, runtime_config_t config);
extern void Slave(runtime_config_t config);

int main(int argc, char **argv) {
  char *input_filename = NULL;
  char *output_filename = NULL;
  int rank, size;
  runtime_config_t config;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (!parse_args(argc, argv, &config, &input_filename, &output_filename)) {
    if (rank == 0) {
      print_usage(argv[0]);
    }
    MPI_Finalize();
    return 1;
  }

  if (rank == 0) {
    printf("Config: mpi=%s, openmp=%s, cuda=%s\n",
           mpi_mode_name(config.mpi_mode),
           openmp_mode_name(config.openmp_mode),
           cuda_mode_name(config.cuda_mode));
  }

  if (rank == 0) {
    Master(input_filename, output_filename, config);
  } else {
    Slave(config);
  }

  MPI_Finalize();
  return 0;
}