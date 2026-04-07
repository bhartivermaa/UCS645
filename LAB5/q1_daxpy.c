// q1_daxpy.c
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define N (1<<16)

int main(int argc, char *argv[]) {
    int rank, size;
    double *X, *Y;
    double a = 2.5;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int local_n = N / size;

    X = (double*) malloc(local_n * sizeof(double));
    Y = (double*) malloc(local_n * sizeof(double));

    for(int i = 0; i < local_n; i++) {
        X[i] = 1.0;
        Y[i] = 2.0;
    }

    double start = MPI_Wtime();

    for(int i = 0; i < local_n; i++)
        X[i] = a * X[i] + Y[i];

    double end = MPI_Wtime();

    if(rank == 0)
        printf("MPI Time: %f\n", end - start);

    free(X);
    free(Y);
    MPI_Finalize();
    return 0;
}
