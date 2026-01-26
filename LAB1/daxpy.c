#include <stdio.h>
#include <omp.h>
#define N 65536

int main() {
    double X[N], Y[N], a = 2.0;

    for (int i = 0; i < N; i++) {
        X[i] = i;
        Y[i] = i;
    }

    double start = omp_get_wtime();

    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        X[i] = a * X[i] + Y[i];
    }

    double end = omp_get_wtime();
    printf("Time: %f seconds\n", end - start);
    return 0;
}
