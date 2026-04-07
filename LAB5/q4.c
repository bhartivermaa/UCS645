// q4_primes.c
#include <mpi.h>
#include <stdio.h>
#include <math.h>

// Function to check prime
int isPrime(int n) {
    if (n < 2) return 0;
    for (int i = 2; i <= sqrt(n); i++) {
        if (n % i == 0) return 0;
    }
    return 1;
}

int main(int argc, char *argv[]) {
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int max = 100; // You can increase this

    if (rank == 0) {
        int num = 2;
        int active = size - 1;

        while (active > 0) {
            int result;
            MPI_Status status;

            // Receive result from any worker
            MPI_Recv(&result, 1, MPI_INT, MPI_ANY_SOURCE, 0,
                     MPI_COMM_WORLD, &status);

            // Assign next number
            if (num <= max) {
                MPI_Send(&num, 1, MPI_INT, status.MPI_SOURCE, 0,
                         MPI_COMM_WORLD);
                num++;
            } else {
                int stop = -1;
                MPI_Send(&stop, 1, MPI_INT, status.MPI_SOURCE, 0,
                         MPI_COMM_WORLD);
                active--;
            }

            // Print only prime numbers
            if (result > 0) {
                printf("Prime number found: %d\n", result);
            }
        }

    } else {
        int request = 0;

        // Initial request
        MPI_Send(&request, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);

        while (1) {
            int num;

            // Receive number to test
            MPI_Recv(&num, 1, MPI_INT, 0, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            if (num < 0) break; // stop signal

            int result;
            if (isPrime(num))
                result = num;
            else
                result = -num;

            // Send result back
            MPI_Send(&result, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        }
    }

    MPI_Finalize();
    return 0;
}
