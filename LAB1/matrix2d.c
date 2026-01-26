#include <stdio.h>
#include <omp.h>
#define N 500

int main() {
    static int A[N][N], B[N][N], C[N][N];

    double start = omp_get_wtime();

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            C[i][j] = 0;
            for (int k = 0; k < N; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    double end = omp_get_wtime();
    printf("2D Matrix Time: %f\n", end - start);
    return 0;
}
