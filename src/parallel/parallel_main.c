#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

extern void Master(char *input_file, char *output_file);
extern void Slave(void);

int main(int argc, char **argv) {
  char *input_filename;
  char *output_filename;
  int rank, size;

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

  if (rank == 0) {
    Master(input_filename, output_filename);
  } else {
    Slave();
  }

  MPI_Finalize();
  return 0;
}
