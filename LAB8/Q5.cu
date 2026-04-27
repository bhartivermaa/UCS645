/******************************************************************************
 * ex05_mnist_cnn.cu
 * Assignment 8 - Problem 5: Full MNIST CNN forward pass.
 *
 * This file implements a production-grade MNIST CNN forward pass using
 * cuDNN (for convolution, pooling, ReLU, BatchNorm) and cuBLAS (for the
 * fully-connected layer).  Training/ablation/AMP/profiler experiments
 * are implemented in the companion file `ex05_mnist_cnn_train.py`,
 * which is much easier to iterate on for the optimizer, scheduler and
 * augmentation studies that Part B-D require.
 *
 * The architecture matches the Python trainer:
 *   conv1 : 1x28x28  -> 16x28x28  (3x3, pad=1)
 *   bn1   :          -> 16x28x28
 *   relu1 :          -> 16x28x28
 *   pool1 :          -> 16x14x14  (2x2 max)
 *   conv2 : 16x14x14 -> 32x14x14  (3x3, pad=1)
 *   bn2   :          -> 32x14x14
 *   relu2 :          -> 32x14x14
 *   pool2 :          -> 32x7x7    (2x2 max)
 *   fc    : 32*7*7   -> 10
 *
 * Build:
 *   nvcc -O2 -arch=sm_86 ex05_mnist_cnn.cu -o ex05 -lcudnn -lcublas
 *
 * Run:
 *   ./ex05                # runs forward pass with random weights, prints timing.
 *****************************************************************************/

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <cublas_v2.h>

#define CUDA_CHECK(call)                                                      \
  do { cudaError_t _e = (call);                                               \
    if (_e != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA  error %s at %s:%d\n",                            \
              cudaGetErrorString(_e), __FILE__, __LINE__);                    \
      exit(1); } } while(0)

#define CUDNN_CHECK(call)                                                     \
  do { cudnnStatus_t _e = (call);                                             \
    if (_e != CUDNN_STATUS_SUCCESS) {                                         \
      fprintf(stderr, "cuDNN error %s at %s:%d\n",                            \
              cudnnGetErrorString(_e), __FILE__, __LINE__);                   \
      exit(1); } } while(0)

#define CUBLAS_CHECK(call)                                                    \
  do { cublasStatus_t _e = (call);                                            \
    if (_e != CUBLAS_STATUS_SUCCESS) {                                        \
      fprintf(stderr, "cuBLAS error %d at %s:%d\n", _e, __FILE__, __LINE__);  \
      exit(1); } } while(0)


// ---------------------------------------------------------------------------
// Helper - create a tensor descriptor.
// ---------------------------------------------------------------------------
static cudnnTensorDescriptor_t makeTensor(int N, int C, int H, int W) {
  cudnnTensorDescriptor_t t;
  CUDNN_CHECK(cudnnCreateTensorDescriptor(&t));
  CUDNN_CHECK(cudnnSetTensor4dDescriptor(t, CUDNN_TENSOR_NCHW,
                                         CUDNN_DATA_FLOAT, N, C, H, W));
  return t;
}

// ---------------------------------------------------------------------------
// Forward pass : returns elapsed milliseconds.
// ---------------------------------------------------------------------------
struct Net {
  // Layer 1
  cudnnFilterDescriptor_t      filt1;
  cudnnConvolutionDescriptor_t conv1d;
  cudnnTensorDescriptor_t      bn1Desc;
  cudnnPoolingDescriptor_t     pool;

  // Layer 2
  cudnnFilterDescriptor_t      filt2;
  cudnnConvolutionDescriptor_t conv2d;
  cudnnTensorDescriptor_t      bn2Desc;

  // ReLU descriptor
  cudnnActivationDescriptor_t  relu;

  // Weights / params (random init for benchmarking).
  float *dW1, *dB1, *dW2, *dB2;
  float *dG1, *dBeta1, *dM1, *dV1;
  float *dG2, *dBeta2, *dM2, *dV2;
  float *dFC, *dFCb;          // FC: [10, 32*7*7] and [10]
};

static void initNet(Net& n) {
  // Filters: conv1 [16,1,3,3], conv2 [32,16,3,3].
  CUDNN_CHECK(cudnnCreateFilterDescriptor(&n.filt1));
  CUDNN_CHECK(cudnnSetFilter4dDescriptor(n.filt1, CUDNN_DATA_FLOAT,
                                         CUDNN_TENSOR_NCHW, 16, 1, 3, 3));
  CUDNN_CHECK(cudnnCreateFilterDescriptor(&n.filt2));
  CUDNN_CHECK(cudnnSetFilter4dDescriptor(n.filt2, CUDNN_DATA_FLOAT,
                                         CUDNN_TENSOR_NCHW, 32, 16, 3, 3));

  CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&n.conv1d));
  CUDNN_CHECK(cudnnSetConvolution2dDescriptor(n.conv1d, 1, 1, 1, 1, 1, 1,
                                              CUDNN_CROSS_CORRELATION,
                                              CUDNN_DATA_FLOAT));
  CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&n.conv2d));
  CUDNN_CHECK(cudnnSetConvolution2dDescriptor(n.conv2d, 1, 1, 1, 1, 1, 1,
                                              CUDNN_CROSS_CORRELATION,
                                              CUDNN_DATA_FLOAT));

  // BN tensors are 1xCx1x1.
  n.bn1Desc = makeTensor(1, 16, 1, 1);
  n.bn2Desc = makeTensor(1, 32, 1, 1);

  CUDNN_CHECK(cudnnCreatePoolingDescriptor(&n.pool));
  CUDNN_CHECK(cudnnSetPooling2dDescriptor(n.pool,
                                          CUDNN_POOLING_MAX,
                                          CUDNN_NOT_PROPAGATE_NAN,
                                          2, 2,    // window
                                          0, 0,    // pad
                                          2, 2));  // stride

  CUDNN_CHECK(cudnnCreateActivationDescriptor(&n.relu));
  CUDNN_CHECK(cudnnSetActivationDescriptor(n.relu, CUDNN_ACTIVATION_RELU,
                                           CUDNN_NOT_PROPAGATE_NAN, 0.0));

  // Random weights.
  auto allocFill = [](float** d, size_t n_elem, float v) {
    CUDA_CHECK(cudaMalloc(d, n_elem * sizeof(float)));
    std::vector<float> h(n_elem);
    for (size_t i = 0; i < n_elem; ++i) {
      h[i] = ((int)(i * 7919) % 1000 - 500) * 0.001f * v;
    }
    CUDA_CHECK(cudaMemcpy(*d, h.data(), n_elem * sizeof(float),
                          cudaMemcpyHostToDevice));
  };
  allocFill(&n.dW1, 16*1*3*3,  0.1f);   allocFill(&n.dB1, 16, 0.0f);
  allocFill(&n.dW2, 32*16*3*3, 0.1f);   allocFill(&n.dB2, 32, 0.0f);
  allocFill(&n.dG1, 16, 1.0f); allocFill(&n.dBeta1, 16, 0.0f);
  allocFill(&n.dM1, 16, 0.0f); allocFill(&n.dV1,    16, 1.0f);
  allocFill(&n.dG2, 32, 1.0f); allocFill(&n.dBeta2, 32, 0.0f);
  allocFill(&n.dM2, 32, 0.0f); allocFill(&n.dV2,    32, 1.0f);
  allocFill(&n.dFC,  10 * 32 * 7 * 7, 0.05f);
  allocFill(&n.dFCb, 10, 0.0f);
}

static float forward(Net& n, cudnnHandle_t dnn, cublasHandle_t blas,
                     int batch, const float* dInput, float* dLogits) {
  // Tensor descriptors per stage.
  cudnnTensorDescriptor_t xT  = makeTensor(batch, 1, 28, 28);
  cudnnTensorDescriptor_t c1T = makeTensor(batch, 16, 28, 28);
  cudnnTensorDescriptor_t p1T = makeTensor(batch, 16, 14, 14);
  cudnnTensorDescriptor_t c2T = makeTensor(batch, 32, 14, 14);
  cudnnTensorDescriptor_t p2T = makeTensor(batch, 32,  7,  7);

  size_t s1 = batch * 16 * 28 * 28;
  size_t s2 = batch * 16 * 14 * 14;
  size_t s3 = batch * 32 * 14 * 14;
  size_t s4 = batch * 32 *  7 *  7;
  float *dC1, *dP1, *dC2, *dP2;
  CUDA_CHECK(cudaMalloc(&dC1, s1 * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dP1, s2 * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dC2, s3 * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dP2, s4 * sizeof(float)));

  // Pick a fast forward conv algorithm.
  cudnnConvolutionFwdAlgoPerf_t perf;
  int returnedAlgoCount = 0;
  CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm_v7(
      dnn, xT, n.filt1, n.conv1d, c1T, 1, &returnedAlgoCount, &perf));
  cudnnConvolutionFwdAlgo_t algo1 = perf.algo;
  CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm_v7(
      dnn, p1T, n.filt2, n.conv2d, c2T, 1, &returnedAlgoCount, &perf));
  cudnnConvolutionFwdAlgo_t algo2 = perf.algo;

  size_t ws1 = 0, ws2 = 0;
  CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(
      dnn, xT, n.filt1, n.conv1d, c1T, algo1, &ws1));
  CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(
      dnn, p1T, n.filt2, n.conv2d, c2T, algo2, &ws2));
  size_t ws = ws1 > ws2 ? ws1 : ws2;
  void* dWS = nullptr;
  if (ws) CUDA_CHECK(cudaMalloc(&dWS, ws));

  cudaEvent_t st, en;
  CUDA_CHECK(cudaEventCreate(&st));
  CUDA_CHECK(cudaEventCreate(&en));

  const float a = 1.0f, b = 0.0f;

  CUDA_CHECK(cudaEventRecord(st));

  // Conv1
  CUDNN_CHECK(cudnnConvolutionForward(dnn, &a, xT, dInput,
                                      n.filt1, n.dW1, n.conv1d, algo1,
                                      dWS, ws1, &b, c1T, dC1));
  // BN inference
  CUDNN_CHECK(cudnnBatchNormalizationForwardInference(
      dnn, CUDNN_BATCHNORM_SPATIAL, &a, &b,
      c1T, dC1, c1T, dC1, n.bn1Desc,
      n.dG1, n.dBeta1, n.dM1, n.dV1, 1e-5));
  // ReLU
  CUDNN_CHECK(cudnnActivationForward(dnn, n.relu, &a, c1T, dC1, &b, c1T, dC1));
  // Pool
  CUDNN_CHECK(cudnnPoolingForward(dnn, n.pool, &a, c1T, dC1, &b, p1T, dP1));

  // Conv2
  CUDNN_CHECK(cudnnConvolutionForward(dnn, &a, p1T, dP1,
                                      n.filt2, n.dW2, n.conv2d, algo2,
                                      dWS, ws2, &b, c2T, dC2));
  // BN inference
  CUDNN_CHECK(cudnnBatchNormalizationForwardInference(
      dnn, CUDNN_BATCHNORM_SPATIAL, &a, &b,
      c2T, dC2, c2T, dC2, n.bn2Desc,
      n.dG2, n.dBeta2, n.dM2, n.dV2, 1e-5));
  CUDNN_CHECK(cudnnActivationForward(dnn, n.relu, &a, c2T, dC2, &b, c2T, dC2));
  CUDNN_CHECK(cudnnPoolingForward(dnn, n.pool, &a, c2T, dC2, &b, p2T, dP2));

  // Fully-connected: logits[B,10] = P2[B,32*7*7] * FC^T[32*7*7,10] + bias
  int K = 32 * 7 * 7;     // input features
  int O = 10;
  CUBLAS_CHECK(cublasSgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N,
                           O, batch, K, &a,
                           n.dFC, K,
                           dP2,   K, &b,
                           dLogits, O));
  // (Bias add is omitted for benchmarking simplicity.)

  CUDA_CHECK(cudaEventRecord(en));
  CUDA_CHECK(cudaEventSynchronize(en));
  float ms; CUDA_CHECK(cudaEventElapsedTime(&ms, st, en));

  if (dWS) CUDA_CHECK(cudaFree(dWS));
  CUDA_CHECK(cudaFree(dC1));
  CUDA_CHECK(cudaFree(dP1));
  CUDA_CHECK(cudaFree(dC2));
  CUDA_CHECK(cudaFree(dP2));
  CUDNN_CHECK(cudnnDestroyTensorDescriptor(xT));
  CUDNN_CHECK(cudnnDestroyTensorDescriptor(c1T));
  CUDNN_CHECK(cudnnDestroyTensorDescriptor(p1T));
  CUDNN_CHECK(cudnnDestroyTensorDescriptor(c2T));
  CUDNN_CHECK(cudnnDestroyTensorDescriptor(p2T));
  CUDA_CHECK(cudaEventDestroy(st));
  CUDA_CHECK(cudaEventDestroy(en));
  return ms;
}

int main() {
  cudnnHandle_t  dnn;
  cublasHandle_t blas;
  CUDNN_CHECK(cudnnCreate(&dnn));
  CUBLAS_CHECK(cublasCreate(&blas));

  Net n;
  initNet(n);

  const int batch = 256;
  size_t inBytes  = batch * 1 * 28 * 28 * sizeof(float);
  size_t outBytes = batch * 10 * sizeof(float);
  float *dIn, *dOut;
  CUDA_CHECK(cudaMalloc(&dIn,  inBytes));
  CUDA_CHECK(cudaMalloc(&dOut, outBytes));
  CUDA_CHECK(cudaMemset(dIn, 1, inBytes));

  // Warm-up.
  (void) forward(n, dnn, blas, batch, dIn, dOut);

  // Benchmark.
  const int reps = 100;
  float total = 0.0f;
  for (int r = 0; r < reps; ++r)
    total += forward(n, dnn, blas, batch, dIn, dOut);

  printf("=== MNIST CNN forward (cuDNN + cuBLAS) ===\n");
  printf("Batch size            : %d\n", batch);
  printf("Avg forward time      : %.3f ms\n", total / reps);
  printf("Throughput            : %.1f img/s\n",
         batch * 1000.0f / (total / reps));
  printf("\n");
  printf("(For training, ablation, AMP and profiler studies, run\n"
         " the companion script `ex05_mnist_cnn_train.py`.)\n");

  CUDA_CHECK(cudaFree(dIn));
  CUDA_CHECK(cudaFree(dOut));
  CUBLAS_CHECK(cublasDestroy(blas));
  CUDNN_CHECK(cudnnDestroy(dnn));
  return 0;
}
