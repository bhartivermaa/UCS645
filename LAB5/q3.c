// q3_dotproduct.c
#include <mpi.h>
#include <stdio.h>

#define N 500000000

int main(int argc, char *argv[]) {
    int rank, size;
    double multiplier = 2.0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    MPI_Bcast(&multiplier, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    long local_n = N / size;
    double local_sum = 0.0;

    double start = MPI_Wtime();

    for(long i = 0; i < local_n; i++) {
        double A = 1.0;
        double B = 2.0 * multiplier;
        local_sum += A * B;
    }

    double total;
    MPI_Reduce(&local_sum, &total, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    double end = MPI_Wtime();

    if(rank == 0){
        printf("Dot Product: %f\n", total);
        printf("Time: %f\n", end-start);
    }

    MPI_Finalize();
}
