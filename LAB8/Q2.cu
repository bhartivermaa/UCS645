/******************************************************************************
 * ex02_memory_hierarchy.cu
 * Assignment 8 - Problem 2: Parallel Reduction & Shared Memory Optimization
 *
 * Topics: shared memory, tree reduction, warp shuffle reduction,
 * bank conflicts, padded shared-memory tile, shared-memory histogram.
 *
 * Build:
 *   nvcc -O2 -arch=sm_86 ex02_memory_hierarchy.cu -o ex02
 *****************************************************************************/

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>

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
// SECTION A : Reference scale kernel.
// ===========================================================================
__global__ void scaleKernel(float* x, float alpha, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) x[idx] *= alpha;
}

// ===========================================================================
// SECTION B : DIY reduction strategies and memory patterns.
// ===========================================================================

// ---------------------------------------------------------------------------
// TODO B1 (Naive baseline): sequential reduction performed by a single
// thread on the GPU. Used purely as a worst-case timing reference.
// ---------------------------------------------------------------------------
__global__ void sumNaiveKernel(const float* in, float* out, int N) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    float s = 0.0f;
    for (int i = 0; i < N; ++i) s += in[i];
    *out = s;
  }
}

// ---------------------------------------------------------------------------
// TODO B2 (Shared-memory tree reduction): each block reduces blockDim.x
// elements into a partial sum.  A second pass (or atomic) accumulates
// per-block partials into the global result.
// HINT:
//   * extern __shared__ float sdata[];
//   * Stage : load -> __syncthreads() -> halving loop -> write partial.
// ---------------------------------------------------------------------------
__global__ void sumSharedKernel(const float* in, float* partial, int N) {
  extern __shared__ float sdata[];
  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  sdata[tid] = (idx < N) ? in[idx] : 0.0f;
  __syncthreads();

  // Halving loop (sequential addressing, no bank conflict).
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) sdata[tid] += sdata[tid + s];
    __syncthreads();
  }
  if (tid == 0) partial[blockIdx.x] = sdata[0];
}

// ---------------------------------------------------------------------------
// Max reduction (companion to sum) using shared memory tree reduction.
// ---------------------------------------------------------------------------
__global__ void maxSharedKernel(const float* in, float* partial, int N) {
  extern __shared__ float sdata[];
  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  sdata[tid] = (idx < N) ? in[idx] : -INFINITY;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
    __syncthreads();
  }
  if (tid == 0) partial[blockIdx.x] = sdata[0];
}

// ---------------------------------------------------------------------------
// TODO B3 (Bank-conflict demo).
//   Each block declares 32x32 shared-memory tiles.  The kernel performs
//   N_iters writes followed by reads at a configurable stride.
//   stride = 1 : conflict-free.  stride = 32 : 32-way conflict.
// ---------------------------------------------------------------------------
template<int STRIDE>
__global__ void bankConflictKernel(float* out, int iters) {
  __shared__ float tile[32 * 32];
  int tid = threadIdx.x;            // 0..1023 if blockDim.x=1024
  // Map tid to a (row,col) such that consecutive tids access shared mem
  // with the requested stride, which is what causes the conflicts.
  int idx = (tid * STRIDE) & (32*32 - 1);

  float v = (float) tid;
  for (int i = 0; i < iters; ++i) {
    tile[idx] = v;
    __syncthreads();
    v += tile[idx];
    __syncthreads();
  }
  if (tid == 0) out[blockIdx.x] = v;
}

// ---------------------------------------------------------------------------
// TODO B4 (Histogram).
//   Naive global-memory atomicAdd implementation (slow due to contention).
// ---------------------------------------------------------------------------
__global__ void histogramGlobalKernel(const int* data, int* hist,
                                      int N, int nbins) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    int b = data[idx];
    if (b >= 0 && b < nbins) atomicAdd(&hist[b], 1);
  }
}

// Per-block private histogram in shared memory, then merged into global.
__global__ void histogramSharedKernel(const int* data, int* hist,
                                      int N, int nbins) {
  extern __shared__ int sHist[];
  for (int i = threadIdx.x; i < nbins; i += blockDim.x) sHist[i] = 0;
  __syncthreads();

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = idx; i < N; i += stride) {
    int b = data[i];
    if (b >= 0 && b < nbins) atomicAdd(&sHist[b], 1);
  }
  __syncthreads();

  for (int i = threadIdx.x; i < nbins; i += blockDim.x) {
    if (sHist[i]) atomicAdd(&hist[i], sHist[i]);
  }
}

// ===========================================================================
// SECTION C : Warp-shuffle reduction (stretch).
// ===========================================================================

// ---------------------------------------------------------------------------
// TODO C1 - Warp-level reduction using __shfl_down_sync.
//   * 1 warp (32 threads) reduces 32 values in 5 instructions, no shared mem.
//   * Block reduction = warp-shuffle within each warp, then warp 0
//     combines per-warp results via shared memory.
// ---------------------------------------------------------------------------
__inline__ __device__ float warpReduceSum(float v) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    v += __shfl_down_sync(0xFFFFFFFF, v, offset);
  }
  return v;
}

__global__ void sumShuffleKernel(const float* in, float* partial, int N) {
  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  float v = (idx < N) ? in[idx] : 0.0f;

  v = warpReduceSum(v);                     // intra-warp

  __shared__ float warpSums[32];            // up to 32 warps / block
  int laneId = tid & 31;
  int warpId = tid >> 5;
  if (laneId == 0) warpSums[warpId] = v;
  __syncthreads();

  if (warpId == 0) {
    int numWarps = blockDim.x / 32;
    v = (laneId < numWarps) ? warpSums[laneId] : 0.0f;
    v = warpReduceSum(v);
    if (laneId == 0) partial[blockIdx.x] = v;
  }
}

// ===========================================================================
// Host driver helpers.
// ===========================================================================

float reduceWith(const float* dIn, int N, int tpb,
                 void(*kernel)(const float*, float*, int),
                 float& kernelMs) {
  int blocks = (N + tpb - 1) / tpb;
  float *dPartial;
  CUDA_CHECK(cudaMalloc(&dPartial, blocks * sizeof(float)));

  cudaEvent_t s, e;
  CUDA_CHECK(cudaEventCreate(&s));
  CUDA_CHECK(cudaEventCreate(&e));
  CUDA_CHECK(cudaEventRecord(s));
  kernel<<<blocks, tpb, tpb * sizeof(float)>>>(dIn, dPartial, N);
  CUDA_CHECK(cudaEventRecord(e));
  CUDA_CHECK(cudaEventSynchronize(e));
  CUDA_CHECK(cudaEventElapsedTime(&kernelMs, s, e));

  std::vector<float> hPartial(blocks);
  CUDA_CHECK(cudaMemcpy(hPartial.data(), dPartial,
                        blocks * sizeof(float), cudaMemcpyDeviceToHost));
  double sum = 0.0;
  for (float v : hPartial) sum += v;

  CUDA_CHECK(cudaFree(dPartial));
  CUDA_CHECK(cudaEventDestroy(s));
  CUDA_CHECK(cudaEventDestroy(e));
  return (float) sum;
}

static void partA_threeReductions() {
  printf("=== Part A : Three reduction strategies (N = 2^20) ===\n");
  const int N = 1 << 20;
  size_t bytes = N * sizeof(float);

  std::vector<float> hIn(N);
  double ref = 0.0;
  for (int i = 0; i < N; ++i) {
    hIn[i] = ((i % 7) - 3) * 0.5f;
    ref += hIn[i];
  }
  float *dIn;
  CUDA_CHECK(cudaMalloc(&dIn, bytes));
  CUDA_CHECK(cudaMemcpy(dIn, hIn.data(), bytes, cudaMemcpyHostToDevice));

  // Naive (single thread on GPU - intentionally slow).
  float *dOne;
  CUDA_CHECK(cudaMalloc(&dOne, sizeof(float)));
  cudaEvent_t s, e;
  CUDA_CHECK(cudaEventCreate(&s));
  CUDA_CHECK(cudaEventCreate(&e));
  CUDA_CHECK(cudaEventRecord(s));
  sumNaiveKernel<<<1, 1>>>(dIn, dOne, N);
  CUDA_CHECK(cudaEventRecord(e));
  CUDA_CHECK(cudaEventSynchronize(e));
  float t_naive = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&t_naive, s, e));
  float r_naive = 0.0f;
  CUDA_CHECK(cudaMemcpy(&r_naive, dOne, sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(dOne));

  // Shared-memory tree.
  float t_shared = 0.0f;
  float r_shared = reduceWith(dIn, N, 256, sumSharedKernel, t_shared);

  // Warp shuffle.
  float t_shfl = 0.0f;
  float r_shfl = reduceWith(dIn, N, 256, sumShuffleKernel, t_shfl);

  auto bandwidth = [&](float ms) {
    return (bytes / 1.0e9f) / (ms / 1000.0f);   // GB/s read
  };

  printf("%-12s %-12s %-12s %-12s %-12s\n",
         "strategy", "time(us)", "GB/s", "result", "abs_err");
  printf("%-12s %-12.1f %-12.2f %-12.4f %-12.2e\n",
         "naive",   t_naive  * 1000.0f, bandwidth(t_naive),  r_naive,
         std::fabs(r_naive  - (float)ref));
  printf("%-12s %-12.1f %-12.2f %-12.4f %-12.2e\n",
         "shared",  t_shared * 1000.0f, bandwidth(t_shared), r_shared,
         std::fabs(r_shared - (float)ref));
  printf("%-12s %-12.1f %-12.2f %-12.4f %-12.2e\n",
         "shuffle", t_shfl   * 1000.0f, bandwidth(t_shfl),   r_shfl,
         std::fabs(r_shfl   - (float)ref));
  printf("Reference (CPU double) : %.4f\n\n", ref);

  CUDA_CHECK(cudaFree(dIn));
  CUDA_CHECK(cudaEventDestroy(s));
  CUDA_CHECK(cudaEventDestroy(e));
}

template<int STRIDE>
static float runBankConflict(int blocks, int tpb, int iters) {
  float* dOut;
  CUDA_CHECK(cudaMalloc(&dOut, blocks * sizeof(float)));
  cudaEvent_t s, e;
  CUDA_CHECK(cudaEventCreate(&s));
  CUDA_CHECK(cudaEventCreate(&e));

  // Warm-up
  bankConflictKernel<STRIDE><<<blocks, tpb>>>(dOut, 4);
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaEventRecord(s));
  bankConflictKernel<STRIDE><<<blocks, tpb>>>(dOut, iters);
  CUDA_CHECK(cudaEventRecord(e));
  CUDA_CHECK(cudaEventSynchronize(e));
  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, s, e));
  CUDA_CHECK(cudaFree(dOut));
  CUDA_CHECK(cudaEventDestroy(s));
  CUDA_CHECK(cudaEventDestroy(e));
  return ms;
}

static void partB_bankConflicts() {
  printf("=== Part B : Bank-conflict timing ===\n");
  const int blocks = 1024;
  const int tpb    = 1024;
  const int iters  = 200;
  printf("%-10s %-12s\n", "stride", "time(ms)");
  printf("%-10d %-12.4f\n", 1,  runBankConflict<1 >(blocks, tpb, iters));
  printf("%-10d %-12.4f\n", 2,  runBankConflict<2 >(blocks, tpb, iters));
  printf("%-10d %-12.4f\n", 4,  runBankConflict<4 >(blocks, tpb, iters));
  printf("%-10d %-12.4f\n", 8,  runBankConflict<8 >(blocks, tpb, iters));
  printf("%-10d %-12.4f\n", 16, runBankConflict<16>(blocks, tpb, iters));
  printf("%-10d %-12.4f\n", 32, runBankConflict<32>(blocks, tpb, iters));
  printf("\n");
}

// 2D shared-memory tile : transpose with and without padding.
__global__ void transposeNaive(const float* in, float* out, int W) {
  __shared__ float tile[16][16];
  int x = blockIdx.x * 16 + threadIdx.x;
  int y = blockIdx.y * 16 + threadIdx.y;
  tile[threadIdx.y][threadIdx.x] = in[y * W + x];
  __syncthreads();
  int xt = blockIdx.y * 16 + threadIdx.x;
  int yt = blockIdx.x * 16 + threadIdx.y;
  out[yt * W + xt] = tile[threadIdx.x][threadIdx.y];   // bank-conflict read
}

__global__ void transposePadded(const float* in, float* out, int W) {
  __shared__ float tile[16][17];   // +1 padding eliminates conflict
  int x = blockIdx.x * 16 + threadIdx.x;
  int y = blockIdx.y * 16 + threadIdx.y;
  tile[threadIdx.y][threadIdx.x] = in[y * W + x];
  __syncthreads();
  int xt = blockIdx.y * 16 + threadIdx.x;
  int yt = blockIdx.x * 16 + threadIdx.y;
  out[yt * W + xt] = tile[threadIdx.x][threadIdx.y];
}

static void partB_padding() {
  printf("=== Part B : Padded vs un-padded shared tile (transpose) ===\n");
  const int W = 4096;
  size_t bytes = (size_t) W * W * sizeof(float);
  float *dIn, *dOut;
  CUDA_CHECK(cudaMalloc(&dIn,  bytes));
  CUDA_CHECK(cudaMalloc(&dOut, bytes));
  CUDA_CHECK(cudaMemset(dIn, 1, bytes));

  dim3 block(16, 16);
  dim3 grid(W / 16, W / 16);

  cudaEvent_t s, e;
  CUDA_CHECK(cudaEventCreate(&s));
  CUDA_CHECK(cudaEventCreate(&e));
  // Warm-ups.
  transposeNaive <<<grid, block>>>(dIn, dOut, W);
  transposePadded<<<grid, block>>>(dIn, dOut, W);
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaEventRecord(s));
  for (int i = 0; i < 10; ++i) transposeNaive<<<grid, block>>>(dIn, dOut, W);
  CUDA_CHECK(cudaEventRecord(e));
  CUDA_CHECK(cudaEventSynchronize(e));
  float t_naive = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&t_naive, s, e));
  t_naive /= 10.0f;

  CUDA_CHECK(cudaEventRecord(s));
  for (int i = 0; i < 10; ++i) transposePadded<<<grid, block>>>(dIn, dOut, W);
  CUDA_CHECK(cudaEventRecord(e));
  CUDA_CHECK(cudaEventSynchronize(e));
  float t_pad = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&t_pad, s, e));
  t_pad /= 10.0f;

  printf("Un-padded transpose : %.4f ms\n", t_naive);
  printf("Padded   transpose : %.4f ms\n", t_pad);
  printf("Speedup            : %.2fx\n\n", t_naive / t_pad);

  CUDA_CHECK(cudaFree(dIn));
  CUDA_CHECK(cudaFree(dOut));
  CUDA_CHECK(cudaEventDestroy(s));
  CUDA_CHECK(cudaEventDestroy(e));
}

static void partC_histogram() {
  printf("=== Part C : Histogram (global-atomic vs shared-private) ===\n");
  const int N     = 1 << 22;
  const int nbins = 256;

  std::vector<int> hData(N);
  std::vector<int> hRef(nbins, 0);
  for (int i = 0; i < N; ++i) {
    hData[i] = (i * 1664525 + 1013904223) & (nbins - 1);
    hRef[hData[i]]++;
  }

  int *dData, *dHist;
  CUDA_CHECK(cudaMalloc(&dData, N * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&dHist, nbins * sizeof(int)));
  CUDA_CHECK(cudaMemcpy(dData, hData.data(), N * sizeof(int),
                        cudaMemcpyHostToDevice));

  cudaEvent_t s, e;
  CUDA_CHECK(cudaEventCreate(&s));
  CUDA_CHECK(cudaEventCreate(&e));

  CUDA_CHECK(cudaMemset(dHist, 0, nbins * sizeof(int)));
  CUDA_CHECK(cudaEventRecord(s));
  histogramGlobalKernel<<<(N + 255) / 256, 256>>>(dData, dHist, N, nbins);
  CUDA_CHECK(cudaEventRecord(e));
  CUDA_CHECK(cudaEventSynchronize(e));
  float t_glb = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&t_glb, s, e));

  std::vector<int> hHist(nbins);
  CUDA_CHECK(cudaMemcpy(hHist.data(), dHist, nbins * sizeof(int),
                        cudaMemcpyDeviceToHost));
  bool ok_glb = (hHist == hRef);

  CUDA_CHECK(cudaMemset(dHist, 0, nbins * sizeof(int)));
  int blocks = 256;     // grid-stride
  CUDA_CHECK(cudaEventRecord(s));
  histogramSharedKernel<<<blocks, 256, nbins * sizeof(int)>>>
      (dData, dHist, N, nbins);
  CUDA_CHECK(cudaEventRecord(e));
  CUDA_CHECK(cudaEventSynchronize(e));
  float t_sh = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&t_sh, s, e));
  CUDA_CHECK(cudaMemcpy(hHist.data(), dHist, nbins * sizeof(int),
                        cudaMemcpyDeviceToHost));
  bool ok_sh = (hHist == hRef);

  printf("Global atomicAdd : %.4f ms (%s)\n", t_glb, ok_glb ? "OK" : "FAIL");
  printf("Shared private   : %.4f ms (%s)\n", t_sh,  ok_sh  ? "OK" : "FAIL");
  printf("Speedup          : %.2fx\n\n", t_glb / t_sh);

  CUDA_CHECK(cudaFree(dData));
  CUDA_CHECK(cudaFree(dHist));
  CUDA_CHECK(cudaEventDestroy(s));
  CUDA_CHECK(cudaEventDestroy(e));
}

int main() {
  partA_threeReductions();
  partB_bankConflicts();
  partB_padding();
  partC_histogram();
  return 0;
}
