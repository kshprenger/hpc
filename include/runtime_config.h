#ifndef RUNTIME_CONFIG_H
#define RUNTIME_CONFIG_H
typedef enum {
  MPI_MODE_OFF,
  MPI_MODE_AUTO,
  MPI_MODE_FULL,
  MPI_MODE_HYBRID
} mpi_mode_t;

typedef enum {
  OPENMP_MODE_OFF,
  OPENMP_MODE_AUTO,
  OPENMP_MODE_FORCE
} openmp_mode_t;

typedef enum {
  CUDA_MODE_OFF,
  CUDA_MODE_AUTO,
  CUDA_MODE_FORCE
} cuda_mode_t;

typedef struct {
  mpi_mode_t mpi_mode;
  openmp_mode_t openmp_mode;
  cuda_mode_t cuda_mode;
} runtime_config_t;

// #define OPENMP_COARSE_THRESHOLD 30
#define OPENMP_THRESHOLD 20000
#define OPENMP_THREADS_THRESHOLD 3
#define GHOST_OPENMP_THRESHOLD 200000
#define GHOST_OPENMP_THREADS_THRESHOLD 6
#define CUDA_THRESHOLD 20000000

#endif // RUNTIME_CONFIG_H