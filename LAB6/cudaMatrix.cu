#include <stdio.h>

__global__ void matrixAdd(int *A, int *B, int *C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int size = N * N;

    if (idx < size) {
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    int N = 1024;  // large matrix (1024 x 1024)
    int size = N * N * sizeof(int);

    // Host memory
    int *h_A = (int*)malloc(size);
    int *h_B = (int*)malloc(size);
    int *h_C = (int*)malloc(size);

    // Initialize matrices
    for (int i = 0; i < N*N; i++) {
        h_A[i] = 1;
        h_B[i] = 2;
    }

    // Device memory
    int *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // Copy to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Kernel configuration
    int threads = 256;
    int blocks = (N*N + threads - 1) / threads;

    // Launch kernel
    matrixAdd<<<blocks, threads>>>(d_A, d_B, d_C, N);

    // Copy result back
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Print sample output
    printf("Sample Output C[0] = %d\n", h_C[0]);
    printf("Sample Output C[0] = %d\n", h_C[100]);

    // Free memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
