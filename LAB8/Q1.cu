/******************************************************************************
 * ex01_cuda_basics.cu
 * Assignment 8 - Problem 1: GPU Architecture & CUDA Kernel Profiling
 *
 * Topics: thread hierarchy, kernel launch configuration, host-device
 * memory transfers, bandwidth measurement, CPU vs GPU comparison,
 * warp divergence experiment.
 *
 * Build:
 *   nvcc -O2 -arch=sm_86 ex01_cuda_basics.cu -o ex01
 *
 * Run:
 *   ./ex01
 *****************************************************************************/

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <vector>
#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
//  Error-checking macro - wraps every CUDA Runtime call.
// ---------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                      \
  do {                                                                        \
    cudaError_t _err = (call);                                                \
    if (_err != cudaSuccess) {                                                \
      fprintf(stderr, "CUDA error %s at %s:%d - %s\n",                        \
              cudaGetErrorString(_err), __FILE__, __LINE__, #call);           \
      exit(EXIT_FAILURE);                                                     \
    }                                                                         \
  } while (0)

// ===========================================================================
// SECTION A : Reference kernels - PROVIDED.
// ===========================================================================

// Element-wise vector addition: C = A + B.
__global__ void vectorAdd(const float* __restrict__ A,
                          const float* __restrict__ B,
                          float* __restrict__ C,
                          int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    C[idx] = A[idx] + B[idx];
  }
}

// Hello-thread kernel: prints thread coordinates from the GPU.
__global__ void helloThread() {
  printf("block (%d,%d,%d) thread (%d,%d,%d)\n",
         blockIdx.x, blockIdx.y, blockIdx.z,
         threadIdx.x, threadIdx.y, threadIdx.z);
}

// CPU reference for vectorAdd timing comparison.
void cpuVectorAdd(const float* A, const float* B, float* C, int N) {
  for (int i = 0; i < N; ++i) C[i] = A[i] + B[i];
}

// ===========================================================================
// SECTION B : DIY exercises.
// ===========================================================================

// ---------------------------------------------------------------------------
// TODO B1 (DIY): Saxpy kernel  Y = alpha * X + Y.
// HINT:
//   * One thread per element with bounds check.
//   * Signature mirrors vectorAdd.
// ---------------------------------------------------------------------------
__global__ void saxpyKernel(float alpha,
                            const float* __restrict__ X,
                            float* __restrict__ Y,
                            int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    Y[idx] = alpha * X[idx] + Y[idx];
  }
}

// ---------------------------------------------------------------------------
// TODO B2 (DIY): Element-wise squared-difference kernel  out = (a - b)^2.
// HINT:
//   * Useful for MSE building blocks in later exercises.
// ---------------------------------------------------------------------------
__global__ void squaredDiffKernel(const float* __restrict__ A,
                                  const float* __restrict__ B,
                                  float* __restrict__ Out,
                                  int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    float d = A[idx] - B[idx];
    Out[idx] = d * d;
  }
}

// ---------------------------------------------------------------------------
// TODO B3 (DIY): Launch configuration helper.
//   For a given threads_per_block (tpb) compute the number of blocks
//   required to cover N elements with the standard ceil-divide formula.
// HINT:
//   blocks = (N + tpb - 1) / tpb.
// ---------------------------------------------------------------------------
inline int computeBlocks(int N, int tpb) {
  return (N + tpb - 1) / tpb;
}

// Time vectorAdd for one threads_per_block setting and one N.
// Returns elapsed milliseconds (kernel only).
float timeVectorAdd(const float* dA, const float* dB, float* dC,
                    int N, int tpb) {
  int blocks = computeBlocks(N, tpb);

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  // Warm-up.
  vectorAdd<<<blocks, tpb>>>(dA, dB, dC, N);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < 10; ++i) {
    vectorAdd<<<blocks, tpb>>>(dA, dB, dC, N);
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return ms / 10.0f;
}

// ---------------------------------------------------------------------------
// TODO B4 (DIY): Memory bandwidth measurement.
//   Allocate `bytes` on host and device, time H2D and D2H transfers,
//   return achieved GB/s for each direction.
// HINT:
//   * Use cudaEvent for timing.
//   * Use page-locked (pinned) host memory for max throughput.
//   * GB/s = (bytes / 1e9) / (time_ms / 1000).
// ---------------------------------------------------------------------------
struct BandwidthResult {
  size_t  bytes;
  float   h2d_ms;
  float   d2h_ms;
  float   h2d_gbs;
  float   d2h_gbs;
};

BandwidthResult measureBandwidth(size_t bytes) {
  void* hPtr = nullptr;
  void* dPtr = nullptr;
  CUDA_CHECK(cudaMallocHost(&hPtr, bytes));   // pinned host mem
  CUDA_CHECK(cudaMalloc(&dPtr, bytes));
  std::memset(hPtr, 0xA5, bytes);

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  const int reps = 5;

  // H2D
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < reps; ++i) {
    CUDA_CHECK(cudaMemcpy(dPtr, hPtr, bytes, cudaMemcpyHostToDevice));
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float h2d_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&h2d_ms, start, stop));
  h2d_ms /= reps;

  // D2H
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < reps; ++i) {
    CUDA_CHECK(cudaMemcpy(hPtr, dPtr, bytes, cudaMemcpyDeviceToHost));
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float d2h_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&d2h_ms, start, stop));
  d2h_ms /= reps;

  BandwidthResult r;
  r.bytes   = bytes;
  r.h2d_ms  = h2d_ms;
  r.d2h_ms  = d2h_ms;
  r.h2d_gbs = (bytes / 1.0e9f) / (h2d_ms / 1000.0f);
  r.d2h_gbs = (bytes / 1.0e9f) / (d2h_ms / 1000.0f);

  CUDA_CHECK(cudaFree(dPtr));
  CUDA_CHECK(cudaFreeHost(hPtr));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return r;
}

// ===========================================================================
// SECTION C : Stretch goals.
// ===========================================================================

// ---------------------------------------------------------------------------
// Warp divergence experiment.
//   Kernel A : if (threadIdx.x % 2 == 0)  -- intra-warp divergence.
//   Kernel B : if (idx < N/2)             -- divergence only at one warp.
// ---------------------------------------------------------------------------
__global__ void divergentKernel(const float* in, float* out, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N) return;
  float v = in[idx];
  // Force ALU work; the two paths are intentionally different so the
  // hardware cannot collapse them.
  if ((threadIdx.x & 1) == 0) {
    for (int k = 0; k < 32; ++k) v = v * 1.0001f + 0.5f;
  } else {
    for (int k = 0; k < 32; ++k) v = v * 0.9999f - 0.5f;
  }
  out[idx] = v;
}

__global__ void uniformKernel(const float* in, float* out, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N) return;
  float v = in[idx];
  // Same number of FLOPs, no per-thread branch.
  for (int k = 0; k < 32; ++k) v = v * 1.0001f + 0.5f;
  out[idx] = v;
}

// ===========================================================================
// Host driver.
// ===========================================================================

static void printDeviceProps() {
  int dev = 0;
  cudaDeviceProp p;
  CUDA_CHECK(cudaGetDeviceProperties(&p, dev));
  printf("=== Device %d : %s ===\n", dev, p.name);
  printf("  Compute capability      : %d.%d\n", p.major, p.minor);
  printf("  SM count                : %d\n", p.multiProcessorCount);
  printf("  Max threads / block     : %d\n", p.maxThreadsPerBlock);
  printf("  Warp size               : %d\n", p.warpSize);
  printf("  Global memory           : %.2f GiB\n", p.totalGlobalMem / (1024.0 * 1024 * 1024));
  printf("  Shared mem / block      : %zu KiB\n", p.sharedMemPerBlock / 1024);
  printf("  Memory bus width        : %d bits\n", p.memoryBusWidth);

  // memoryClockRate was removed from cudaDeviceProp in CUDA 13.
  // Query it via cudaDeviceGetAttribute, which works on every CUDA version.
  int memClockKHz = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&memClockKHz,
                                    cudaDevAttrMemoryClockRate, dev));
  printf("  Memory clock rate       : %.0f MHz\n", memClockKHz / 1000.0);
  double peak_bw = 2.0 * memClockKHz * 1000.0 * p.memoryBusWidth / 8.0 / 1.0e9;
  printf("  Theoretical peak BW     : %.1f GB/s\n", peak_bw);
  printf("\n");
}

static void partA_speedupSweep() {
  printf("=== Part A : CPU vs GPU sweep ===\n");
  printf("%-12s %-14s %-14s %-14s %-10s\n",
         "N", "CPU(ms)", "GPU_kern(ms)", "H2D(ms)", "Speedup");

  const int Ns[] = {1<<10, 1<<14, 1<<18, 1<<22, 1<<26};
  for (int N : Ns) {
    size_t bytes = N * sizeof(float);

    float* hA = (float*) malloc(bytes);
    float* hB = (float*) malloc(bytes);
    float* hC = (float*) malloc(bytes);
    for (int i = 0; i < N; ++i) { hA[i] = 1.0f; hB[i] = 2.0f; }

    // CPU timing.
    auto t0 = std::chrono::high_resolution_clock::now();
    cpuVectorAdd(hA, hB, hC, N);
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // GPU timing.
    float *dA, *dB, *dC;
    CUDA_CHECK(cudaMalloc(&dA, bytes));
    CUDA_CHECK(cudaMalloc(&dB, bytes));
    CUDA_CHECK(cudaMalloc(&dC, bytes));

    cudaEvent_t s, e;
    CUDA_CHECK(cudaEventCreate(&s));
    CUDA_CHECK(cudaEventCreate(&e));

    CUDA_CHECK(cudaEventRecord(s));
    CUDA_CHECK(cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(e));
    CUDA_CHECK(cudaEventSynchronize(e));
    float h2d_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&h2d_ms, s, e));

    int tpb = 256;
    int blocks = computeBlocks(N, tpb);
    float kern_ms = timeVectorAdd(dA, dB, dC, N, tpb);
    (void)blocks;

    float speedup = (float) cpu_ms / kern_ms;
    printf("%-12d %-14.4f %-14.4f %-14.4f %-10.2f\n",
           N, cpu_ms, kern_ms, h2d_ms, speedup);

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
    free(hA); free(hB); free(hC);
    CUDA_CHECK(cudaEventDestroy(s));
    CUDA_CHECK(cudaEventDestroy(e));
  }
  printf("\n");
}

static void partA_bandwidth() {
  printf("=== Part A : Bandwidth sweep (pinned host memory) ===\n");
  printf("%-10s %-12s %-12s %-12s %-12s\n",
         "MB", "H2D(ms)", "D2H(ms)", "H2D(GB/s)", "D2H(GB/s)");

  const size_t MB = 1024UL * 1024UL;
  const size_t sizes[] = {1*MB, 8*MB, 64*MB, 256*MB, 512*MB};
  for (size_t b : sizes) {
    BandwidthResult r = measureBandwidth(b);
    printf("%-10zu %-12.3f %-12.3f %-12.2f %-12.2f\n",
           b / MB, r.h2d_ms, r.d2h_ms, r.h2d_gbs, r.d2h_gbs);
  }
  printf("\n");
}

static void partB_launchConfig() {
  printf("=== Part B : Launch-configuration sweep (N = 2^20) ===\n");
  const int N = 1 << 20;
  size_t bytes = N * sizeof(float);

  float *dA, *dB, *dC;
 CUDA_CHECK(cudaMalloc(&dA, bytes));
  CUDA_CHECK(cudaMalloc(&dB, bytes));
  CUDA_CHECK(cudaMalloc(&dC, bytes));
  CUDA_CHECK(cudaMemset(dA, 1, bytes));
  CUDA_CHECK(cudaMemset(dB, 2, bytes));

  printf("%-8s %-10s %-12s\n", "tpb", "blocks", "time(ms)");
  const int tpbs[] = {64, 128, 256, 512, 1024};
  for (int tpb : tpbs) {
    int blocks = computeBlocks(N, tpb);
    float ms = timeVectorAdd(dA, dB, dC, N, tpb);
    printf("%-8d %-10d %-12.4f\n", tpb, blocks, ms);
  }
  CUDA_CHECK(cudaFree(dA));
  CUDA_CHECK(cudaFree(dB));
  CUDA_CHECK(cudaFree(dC));
  printf("\n");
}

static void partC_warpDivergence() {
  printf("=== Part C : Warp-divergence experiment ===\n");
  const int N = 1 << 22;
  size_t bytes = N * sizeof(float);

  float *dIn, *dOut;
  CUDA_CHECK(cudaMalloc(&dIn,  bytes));
  CUDA_CHECK(cudaMalloc(&dOut, bytes));
  CUDA_CHECK(cudaMemset(dIn, 0, bytes));

  int tpb = 256;
  int blocks = computeBlocks(N, tpb);

  cudaEvent_t s, e;
  CUDA_CHECK(cudaEventCreate(&s));
  CUDA_CHECK(cudaEventCreate(&e));

  // Warm-up.
  divergentKernel<<<blocks, tpb>>>(dIn, dOut, N);
  uniformKernel  <<<blocks, tpb>>>(dIn, dOut, N);
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaEventRecord(s));
  for (int i = 0; i < 20; ++i)
    divergentKernel<<<blocks, tpb>>>(dIn, dOut, N);
  CUDA_CHECK(cudaEventRecord(e));
  CUDA_CHECK(cudaEventSynchronize(e));
  float t_div = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&t_div, s, e));
  t_div /= 20.0f;

  CUDA_CHECK(cudaEventRecord(s));
  for (int i = 0; i < 20; ++i)
    uniformKernel<<<blocks, tpb>>>(dIn, dOut, N);
  CUDA_CHECK(cudaEventRecord(e));
  CUDA_CHECK(cudaEventSynchronize(e));
  float t_uni = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&t_uni, s, e));
  t_uni /= 20.0f;

  printf("Divergent kernel time : %.4f ms\n", t_div);
  printf("Uniform   kernel time : %.4f ms\n", t_uni);
  printf("Divergence penalty    : %.2fx slower\n\n", t_div / t_uni);

  CUDA_CHECK(cudaFree(dIn));
  CUDA_CHECK(cudaFree(dOut));
  CUDA_CHECK(cudaEventDestroy(s));
  CUDA_CHECK(cudaEventDestroy(e));
}

int main() {
  printDeviceProps();

  // Quick sanity print: 2 blocks of 4 threads.
  printf("=== helloThread (2x4) ===\n");
  helloThread<<<2, 4>>>();
  CUDA_CHECK(cudaDeviceSynchronize());
  printf("\n");

  partA_speedupSweep();
  partA_bandwidth();
  partB_launchConfig();
  partC_warpDivergence();

  printf("Done.\n");
  return 0;
}
