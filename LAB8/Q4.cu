/******************************************************************************
 * ex04_cnn_layers.cu
 * Assignment 8 - Problem 4: Tiled GEMM vs cuBLAS & CNN Layer Benchmarking.
 *
 * Topics: naive GEMM, tiled GEMM, cuBLAS SGEMM, MaxPool2D,
 *         BatchNorm inference, im2col + GEMM convolution.
 *
 * Build:
 *   nvcc -O2 -arch=sm_86 ex04_cnn_layers.cu -o ex04 -lcublas
 *****************************************************************************/

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CUDA_CHECK(call)                                                      \
  do {                                                                        \
    cudaError_t _err = (call);                                                \
    if (_err != cudaSuccess) {                                                \
      fprintf(stderr, "CUDA error %s at %s:%d - %s\n",                        \
              cudaGetErrorString(_err), __FILE__, __LINE__, #call);           \
      exit(EXIT_FAILURE);                                                     \
    }                                                                         \
  } while (0)

#define CUBLAS_CHECK(call)                                                    \
  do {                                                                        \
    cublasStatus_t _st = (call);                                              \
    if (_st != CUBLAS_STATUS_SUCCESS) {                                       \
      fprintf(stderr, "cuBLAS error %d at %s:%d\n", _st, __FILE__, __LINE__); \
      exit(EXIT_FAILURE);                                                     \
    }                                                                         \
  } while (0)

// ===========================================================================
// SECTION A : Reference naive GEMM (no shared memory).
// C [MxN] = A [MxK] * B [KxN].
// ===========================================================================
__global__ void naiveMatMul(const float* A, const float* B, float* C,
                            int M, int N, int K) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < M && col < N) {
    float s = 0.0f;
    for (int k = 0; k < K; ++k) s += A[row * K + k] * B[k * N + col];
    C[row * N + col] = s;
  }
}

// ===========================================================================
// SECTION B : DIY tiled GEMM and CNN layers.
// ===========================================================================

// ---------------------------------------------------------------------------
// TODO B1 - Tiled matrix multiplication using shared memory.
//   Steps :
//     1) Declare __shared__ As[TILE][TILE], Bs[TILE][TILE].
//     2) Loop over K in TILE-sized chunks.
//     3) Cooperatively load A and B tiles, __syncthreads().
//     4) Multiply-accumulate the tile rows/cols.
//     5) __syncthreads(), advance to next tile.
//     6) Write C[row,col].
// ---------------------------------------------------------------------------
template<int TILE>
__global__ void tiledMatMul(const float* A, const float* B, float* C,
                            int M, int N, int K) {
  __shared__ float As[TILE][TILE];
  __shared__ float Bs[TILE][TILE];

  int row = blockIdx.y * TILE + threadIdx.y;
  int col = blockIdx.x * TILE + threadIdx.x;
  float sum = 0.0f;

  for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {
    int aCol = t * TILE + threadIdx.x;
    int bRow = t * TILE + threadIdx.y;
    As[threadIdx.y][threadIdx.x] = (row < M && aCol < K) ? A[row * K + aCol] : 0.0f;
    Bs[threadIdx.y][threadIdx.x] = (bRow < K && col < N) ? B[bRow * N + col] : 0.0f;
    __syncthreads();

#pragma unroll
    for (int k = 0; k < TILE; ++k) {
      sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
    }
    __syncthreads();
  }
  if (row < M && col < N) C[row * N + col] = sum;
}

// ---------------------------------------------------------------------------
// TODO C1 - MaxPool 2D with stride = pool size (no overlap).
//   Input  : [N, C, H, W].
//   Output : [N, C, H/p, W/p].
// ---------------------------------------------------------------------------
__global__ void maxPool2D(const float* in, float* out,
                          int N, int C, int H, int W, int P) {
  int Ho = H / P, Wo = W / P;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = N * C * Ho * Wo;
  if (idx >= total) return;

  int wo = idx % Wo;          int t = idx / Wo;
  int ho = t   % Ho;              t = t   / Ho;
  int c  = t   % C;
  int n  = t   / C;

  float m = -INFINITY;
  for (int dy = 0; dy < P; ++dy) {
    for (int dx = 0; dx < P; ++dx) {
      int h = ho * P + dy;
      int w = wo * P + dx;
      float v = in[((n * C + c) * H + h) * W + w];
      if (v > m) m = v;
    }
  }
  out[idx] = m;
}

// ---------------------------------------------------------------------------
// TODO C2 - BatchNorm (inference mode, channel-wise).
//   y = gamma * (x - mean) / sqrt(var + eps) + beta.
//   Per-channel scale-bias parameters of length C.
// ---------------------------------------------------------------------------
__global__ void batchNormInference(const float* x, float* y,
                                   const float* mean, const float* var,
                                   const float* gamma, const float* beta,
                                   float eps,
                                   int N, int C, int H, int W) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = N * C * H * W;
  if (idx >= total) return;

  int hw = H * W;
  int c = (idx / hw) % C;
  float invStd = rsqrtf(var[c] + eps);
  y[idx] = gamma[c] * (x[idx] - mean[c]) * invStd + beta[c];
}

// ---------------------------------------------------------------------------
// im2col : [N, C, H, W]  ->  [(N * H_out * W_out), (C * kH * kW)]   (row-major).
//   This layout makes convolution = im2col + GEMM(weight^T).
// ---------------------------------------------------------------------------
__global__ void im2colKernel(const float* x, float* col,
                             int N, int C, int H, int W,
                             int kH, int kW, int pad, int stride,
                             int Ho, int Wo) {
  int idx   = blockIdx.x * blockDim.x + threadIdx.x;
  int total = N * Ho * Wo;
  if (idx >= total) return;

  int wo = idx % Wo;       int t = idx / Wo;
  int ho = t   % Ho;           t = t   / Ho;
  int n  = t;

  int row = idx;          // one row per output position
  int colStride = C * kH * kW;
  for (int c = 0; c < C; ++c) {
    for (int j = 0; j < kH; ++j) {
      for (int i = 0; i < kW; ++i) {
        int h = ho * stride - pad + j;
        int w = wo * stride - pad + i;
        float v = 0.0f;
        if (h >= 0 && h < H && w >= 0 && w < W) {
          v = x[((n * C + c) * H + h) * W + w];
        }
        int kIdx = (c * kH + j) * kW + i;
        col[row * colStride + kIdx] = v;
      }
    }
  }
}

// ===========================================================================
// Host helpers : timing / verification.
// ===========================================================================

static void cpuMatMul(const float* A, const float* B, float* C,
                      int M, int N, int K) {
  for (int i = 0; i < M; ++i)
    for (int j = 0; j < N; ++j) {
      double s = 0.0;
      for (int k = 0; k < K; ++k) s += (double) A[i*K+k] * B[k*N+j];
      C[i*N+j] = (float) s;
    }
}

static void partA_gemmCorrectness() {
  printf("=== Part A : Tiled-GEMM correctness (M=K=N=512) ===\n");
  const int M = 512, N = 512, K = 512;
  size_t bytes = M * N * sizeof(float);

  std::vector<float> hA(M*K), hB(K*N), hC(M*N), hRef(M*N);
  for (int i = 0; i < M*K; ++i) hA[i] = ((i % 13) - 6) * 0.01f;
  for (int i = 0; i < K*N; ++i) hB[i] = ((i % 11) - 5) * 0.01f;
  cpuMatMul(hA.data(), hB.data(), hRef.data(), M, N, K);

  float *dA, *dB, *dC;
  CUDA_CHECK(cudaMalloc(&dA, M*K*sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dB, K*N*sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dC, bytes));
  CUDA_CHECK(cudaMemcpy(dA, hA.data(), M*K*sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB, hB.data(), K*N*sizeof(float), cudaMemcpyHostToDevice));

  dim3 block(16, 16), grid(N/16, M/16);
  tiledMatMul<16><<<grid, block>>>(dA, dB, dC, M, N, K);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(hC.data(), dC, bytes, cudaMemcpyDeviceToHost));

  float maxErr = 0.0f;
  for (int i = 0; i < M*N; ++i) {
    float e = std::fabs(hC[i] - hRef[i]);
    if (e > maxErr) maxErr = e;
  }
  printf("Tiled GEMM max|err| = %.2e (target < 1e-3)  %s\n\n",
         maxErr, maxErr < 1e-3f ? "PASS" : "FAIL");
  CUDA_CHECK(cudaFree(dA));
  CUDA_CHECK(cudaFree(dB));
  CUDA_CHECK(cudaFree(dC));
}

static double gflops(int M, int N, int K, float ms) {
  return (2.0 * M * N * K) / (ms * 1.0e6);   // 2*M*N*K FLOPs / (ms * 1e6)  -> GFLOP/s
}

static void partA_gemmBenchmark(cublasHandle_t cub) {
  printf("=== Part A : GEMM benchmark (square sizes) ===\n");
  printf("%-6s %-12s %-12s %-12s %-10s %-10s %-10s\n",
         "size", "naive(ms)", "tiled(ms)", "cuBLAS(ms)",
         "naive_GF", "tiled_GF", "cuBLAS_GF");
  const int sizes[] = {128, 256, 512, 1024, 2048};
  for (int s : sizes) {
    int M = s, N = s, K = s;
    size_t bA = M*K*sizeof(float);
    size_t bB = K*N*sizeof(float);
    size_t bC = M*N*sizeof(float);

    float *dA, *dB, *dC;
    CUDA_CHECK(cudaMalloc(&dA, bA));
    CUDA_CHECK(cudaMalloc(&dB, bB));
    CUDA_CHECK(cudaMalloc(&dC, bC));
    CUDA_CHECK(cudaMemset(dA, 1, bA));
    CUDA_CHECK(cudaMemset(dB, 1, bB));

    cudaEvent_t st, en;
    CUDA_CHECK(cudaEventCreate(&st));
    CUDA_CHECK(cudaEventCreate(&en));

    // Warm-ups.
    {
      dim3 b(16,16), g((N+15)/16, (M+15)/16);
      naiveMatMul<<<g, b>>>(dA, dB, dC, M, N, K);
      tiledMatMul<16><<<g, b>>>(dA, dB, dC, M, N, K);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // naive
    dim3 b(16,16), g((N+15)/16, (M+15)/16);
    int reps = (s <= 512) ? 5 : 2;
    CUDA_CHECK(cudaEventRecord(st));
    for (int r = 0; r < reps; ++r) naiveMatMul<<<g, b>>>(dA, dB, dC, M, N, K);
    CUDA_CHECK(cudaEventRecord(en));
    CUDA_CHECK(cudaEventSynchronize(en));
    float t_naive; CUDA_CHECK(cudaEventElapsedTime(&t_naive, st, en));
    t_naive /= reps;

    // tiled
    CUDA_CHECK(cudaEventRecord(st));
    for (int r = 0; r < reps; ++r) tiledMatMul<16><<<g, b>>>(dA, dB, dC, M, N, K);
    CUDA_CHECK(cudaEventRecord(en));
    CUDA_CHECK(cudaEventSynchronize(en));
    float t_tiled; CUDA_CHECK(cudaEventElapsedTime(&t_tiled, st, en));
    t_tiled /= reps;

    // cuBLAS sgemm  -- column-major, so compute C^T = B^T * A^T to keep row-major.
    const float a = 1.0f, b0 = 0.0f;
    CUDA_CHECK(cudaEventRecord(st));
    for (int r = 0; r < reps; ++r) {
      CUBLAS_CHECK(cublasSgemm(cub, CUBLAS_OP_N, CUBLAS_OP_N,
                               N, M, K, &a,
                               dB, N,
                               dA, K, &b0,
                               dC, N));
    }
    CUDA_CHECK(cudaEventRecord(en));
    CUDA_CHECK(cudaEventSynchronize(en));
    float t_blas; CUDA_CHECK(cudaEventElapsedTime(&t_blas, st, en));
    t_blas /= reps;

    printf("%-6d %-12.3f %-12.3f %-12.3f %-10.1f %-10.1f %-10.1f\n",
           s, t_naive, t_tiled, t_blas,
           gflops(M,N,K,t_naive), gflops(M,N,K,t_tiled), gflops(M,N,K,t_blas));

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
    CUDA_CHECK(cudaEventDestroy(st));
    CUDA_CHECK(cudaEventDestroy(en));
  }
  printf("\n");
}

static void partB_layerBenchmarks() {
  printf("=== Part B : CNN layer benchmarks ([32,64,14,14] tensors) ===\n");
  const int N = 32, C = 64, H = 14, W = 14;
  size_t bytes = (size_t) N * C * H * W * sizeof(float);
  float *dIn, *dOut;
  CUDA_CHECK(cudaMalloc(&dIn,  bytes));
  CUDA_CHECK(cudaMalloc(&dOut, bytes));
  CUDA_CHECK(cudaMemset(dIn, 1, bytes));

  // BatchNorm
  std::vector<float> hMean(C, 0.5f), hVar(C, 1.0f), hG(C, 1.0f), hB(C, 0.0f);
  float *dMean, *dVar, *dGamma, *dBeta;
  CUDA_CHECK(cudaMalloc(&dMean,  C*sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dVar,   C*sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dGamma, C*sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dBeta,  C*sizeof(float)));
  CUDA_CHECK(cudaMemcpy(dMean,  hMean.data(),  C*sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dVar,   hVar.data(),   C*sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dGamma, hG.data(),     C*sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dBeta,  hB.data(),     C*sizeof(float), cudaMemcpyHostToDevice));

  cudaEvent_t st, en;
  CUDA_CHECK(cudaEventCreate(&st));
  CUDA_CHECK(cudaEventCreate(&en));

  int total = N * C * H * W;
  int tpb = 256, blk = (total + tpb - 1) / tpb;

  // Warm-up.
  batchNormInference<<<blk, tpb>>>(dIn, dOut, dMean, dVar, dGamma, dBeta, 1e-5f, N, C, H, W);
  maxPool2D<<<(N*C*(H/2)*(W/2)+255)/256, 256>>>(dIn, dOut, N, C, H, W, 2);
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaEventRecord(st));
  for (int r = 0; r < 50; ++r)
    batchNormInference<<<blk, tpb>>>(dIn, dOut, dMean, dVar, dGamma, dBeta, 1e-5f, N, C, H, W);
  CUDA_CHECK(cudaEventRecord(en));
  CUDA_CHECK(cudaEventSynchronize(en));
  float t_bn; CUDA_CHECK(cudaEventElapsedTime(&t_bn, st, en)); t_bn /= 50;

  int totMP = N * C * (H/2) * (W/2);
  CUDA_CHECK(cudaEventRecord(st));
  for (int r = 0; r < 50; ++r)
    maxPool2D<<<(totMP+255)/256, 256>>>(dIn, dOut, N, C, H, W, 2);
  CUDA_CHECK(cudaEventRecord(en));
  CUDA_CHECK(cudaEventSynchronize(en));
  float t_mp; CUDA_CHECK(cudaEventElapsedTime(&t_mp, st, en)); t_mp /= 50;

  printf("BatchNorm (inference) : %.4f ms\n", t_bn);
  printf("MaxPool 2x2           : %.4f ms\n", t_mp);
  printf("(Compare to PyTorch's nn.functional.batch_norm / max_pool2d in the report.)\n\n");

  CUDA_CHECK(cudaFree(dIn));
  CUDA_CHECK(cudaFree(dOut));
  CUDA_CHECK(cudaFree(dMean));
  CUDA_CHECK(cudaFree(dVar));
  CUDA_CHECK(cudaFree(dGamma));
  CUDA_CHECK(cudaFree(dBeta));
  CUDA_CHECK(cudaEventDestroy(st));
  CUDA_CHECK(cudaEventDestroy(en));
}

static void partC_im2col(cublasHandle_t cub) {
  printf("=== Part C : im2col + GEMM convolution ===\n");
  const int N=8, C=16, H=14, W=14, kH=3, kW=3, pad=1, stride=1;
  const int Cout = 32;
  const int Ho = (H + 2*pad - kH) / stride + 1;
  const int Wo = (W + 2*pad - kW) / stride + 1;

  size_t inBytes  = (size_t) N*C*H*W * sizeof(float);
  size_t colRows  = (size_t) N * Ho * Wo;
  size_t colCols  = (size_t) C * kH * kW;
  size_t colBytes = colRows * colCols * sizeof(float);
  size_t wBytes   = (size_t) Cout * colCols * sizeof(float);
  size_t outBytes = (size_t) N * Cout * Ho * Wo * sizeof(float);

  float *dIn, *dCol, *dW, *dOut;
  CUDA_CHECK(cudaMalloc(&dIn,  inBytes));
  CUDA_CHECK(cudaMalloc(&dCol, colBytes));
  CUDA_CHECK(cudaMalloc(&dW,   wBytes));
  CUDA_CHECK(cudaMalloc(&dOut, outBytes));
  CUDA_CHECK(cudaMemset(dIn, 1, inBytes));
  CUDA_CHECK(cudaMemset(dW,  1, wBytes));

  cudaEvent_t st, en;
  CUDA_CHECK(cudaEventCreate(&st));
  CUDA_CHECK(cudaEventCreate(&en));

  int totalOut = N * Ho * Wo;
  CUDA_CHECK(cudaEventRecord(st));
  for (int r = 0; r < 20; ++r) {
    im2colKernel<<<(totalOut + 255)/256, 256>>>(
        dIn, dCol, N, C, H, W, kH, kW, pad, stride, Ho, Wo);
    // GEMM: out[N*Ho*Wo, Cout] = col[N*Ho*Wo, C*kH*kW] * W^T[C*kH*kW, Cout]
    const float a = 1.0f, b0 = 0.0f;
    int M = (int) colRows;
    int K = (int) colCols;
    CUBLAS_CHECK(cublasSgemm(cub, CUBLAS_OP_T, CUBLAS_OP_N,
                             Cout, M, K, &a,
                             dW,  K,
                             dCol,K, &b0,
                             dOut,Cout));
  }
  CUDA_CHECK(cudaEventRecord(en));
  CUDA_CHECK(cudaEventSynchronize(en));
  float ms; CUDA_CHECK(cudaEventElapsedTime(&ms, st, en)); ms /= 20;

  double flops_per_call = 2.0 * (double) Cout * colRows * colCols;
  double gf = flops_per_call / (ms * 1.0e6);
  double overhead = (double) colBytes / inBytes;
  printf("im2col+GEMM time    : %.4f ms\n", ms);
  printf("Equivalent GFLOP/s  : %.1f\n", gf);
  printf("Memory blow-up factor (col / input) : %.2fx\n\n", overhead);

  CUDA_CHECK(cudaFree(dIn));
  CUDA_CHECK(cudaFree(dCol));
  CUDA_CHECK(cudaFree(dW));
  CUDA_CHECK(cudaFree(dOut));
  CUDA_CHECK(cudaEventDestroy(st));
  CUDA_CHECK(cudaEventDestroy(en));
}

int main() {
  cublasHandle_t cub;
  CUBLAS_CHECK(cublasCreate(&cub));

  partA_gemmCorrectness();
  partA_gemmBenchmark(cub);
  partB_layerBenchmarks();
  partC_im2col(cub);

  CUBLAS_CHECK(cublasDestroy(cub));
  return 0;
}
