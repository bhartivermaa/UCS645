#include <stdio.h>

__global__ void sumKernel(float *input, float *output, int n) {
    __shared__ float cache[256];

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIndex = threadIdx.x;

    float temp = 0;

    // Each thread processes multiple elements
    while (tid < n) {
        temp += input[tid];
        tid += blockDim.x * gridDim.x;
    }

    cache[cacheIndex] = temp;
    __syncthreads();

    // Reduction in shared memory
    int i = blockDim.x / 2;
    while (i != 0) {
        if (cacheIndex < i)
            cache[cacheIndex] += cache[cacheIndex + i];
        __syncthreads();
        i /= 2;
    }

    // Store result from each block
    if (cacheIndex == 0)
        output[blockIdx.x] = cache[0];
}

int main() {
    int n = 1024;
    size_t size = n * sizeof(float);

    // 🔹 1. Allocate host memory
    float *h_input = (float*)malloc(size);
    float *h_output;

    // Generate input
    for (int i = 0; i < n; i++)
        h_input[i] = 1.0f;

    // 🔹 2. Allocate device memory
    float *d_input, *d_output;
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, sizeof(float) * 4);

    // 🔹 3. Copy host → device
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    // 🔹 4. Define grid & block
    int threads = 256;
    int blocks = 4;

    // 🔹 5. Launch kernel
    sumKernel<<<blocks, threads>>>(d_input, d_output, n);

    // 🔹 6. Copy result back
    h_output = (float*)malloc(sizeof(float) * blocks);
    cudaMemcpy(h_output, d_output, sizeof(float) * blocks, cudaMemcpyDeviceToHost);

    // Final sum on CPU
    float final_sum = 0;
    for (int i = 0; i < blocks; i++)
        final_sum += h_output[i];

    printf("Final Sum = %f\n", final_sum);

    // 🔹 7. Free memory
    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);

    return 0;
}
