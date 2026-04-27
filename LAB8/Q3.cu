/******************************************************************************
 * ex03_ml_primitives.cu
 * Assignment 8 - Problem 3: Custom ML kernels.
 *
 * Topics: activation functions (sigmoid, tanh, leaky-ReLU, ReLU-backward),
 * loss functions (BCE, numerically-stable CE), CE gradient, fused Adam.
 *
 * Build:
 *   nvcc -O2 -arch=sm_86 ex03_ml_primitives.cu -o ex03 -lm
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
// SECTION A : Reference activation - ReLU forward.
// ===========================================================================
__global__ void reluFwd(const float* x, float* y, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) y[i] = fmaxf(0.0f, x[i]);
}

// ===========================================================================
// SECTION B : DIY activation kernels.
// ===========================================================================

// TODO B1 - Sigmoid.   y = 1 / (1 + exp(-x))
__global__ void sigmoidFwd(const float* x, float* y, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) y[i] = 1.0f / (1.0f + expf(-x[i]));
}

// TODO B2 - Tanh.       y = tanhf(x)  (use built-in for numerical stability)
__global__ void tanhFwd(const float* x, float* y, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) y[i] = tanhf(x[i]);
}

// TODO B3 - Leaky ReLU. y = x if x>0 else alpha*x
__global__ void leakyReluFwd(const float* x, float* y, float alpha, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    float v = x[i];
    y[i] = v > 0.0f ? v : alpha * v;
  }
}

// TODO B4 - ReLU backward.  dx = dy * (x > 0 ? 1 : 0)
__global__ void reluBwd(const float* x, const float* dy, float* dx, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) dx[i] = (x[i] > 0.0f) ? dy[i] : 0.0f;
}

// ===========================================================================
// SECTION C : Loss functions.
// ===========================================================================

// ---------------------------------------------------------------------------
// TODO C1 - Binary cross-entropy with logit clipping.
// HINT:
//   * BCE = - [ y*log(p) + (1-y)*log(1-p) ]
//   * Numerically stable form using softplus:
//        max(z,0) - z*y + log(1 + exp(-|z|))
// ---------------------------------------------------------------------------
__global__ void bceLossKernel(const float* logits, const float* labels,
                              float* losses, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    float z = logits[i];
    float y = labels[i];
    float maxz = fmaxf(z, 0.0f);
    losses[i] = maxz - z * y + logf(1.0f + expf(-fabsf(z)));
  }
}

// ---------------------------------------------------------------------------
// TODO C2 - Cross-entropy via log-sum-exp.
//   Each thread handles one row of logits [N, C], with label[N].
//   loss[i] = -logits[i,label[i]] + logsumexp(logits[i, :]).
// HINT:
//   * Subtract row-max before exp() to avoid overflow.
// ---------------------------------------------------------------------------
__global__ void crossEntropyKernel(const float* logits, const int* labels,
                                   float* losses, int N, int C) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;

  const float* row = logits + i * C;
  float maxv = row[0];
  for (int c = 1; c < C; ++c) maxv = fmaxf(maxv, row[c]);

  float sumExp = 0.0f;
  for (int c = 0; c < C; ++c) sumExp += expf(row[c] - maxv);
  float lse = maxv + logf(sumExp);

  int lbl = labels[i];
  losses[i] = -row[lbl] + lse;
}

// ---------------------------------------------------------------------------
// CE gradient w.r.t. logits.
//   grad[i,c] = softmax(logits[i,:])[c] - one_hot(label[i])[c].
//   Output is averaged over the batch by dividing by N.
// ---------------------------------------------------------------------------
__global__ void crossEntropyGradKernel(const float* logits,
                                       const int*   labels,
                                       float*       grad,
                                       int N, int C) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;

  const float* row = logits + i * C;
  float maxv = row[0];
  for (int c = 1; c < C; ++c) maxv = fmaxf(maxv, row[c]);
  float sumExp = 0.0f;
  for (int c = 0; c < C; ++c) sumExp += expf(row[c] - maxv);

  int lbl = labels[i];
  float invN = 1.0f / (float) N;
  for (int c = 0; c < C; ++c) {
    float p = expf(row[c] - maxv) / sumExp;
    grad[i * C + c] = (p - (c == lbl ? 1.0f : 0.0f)) * invN;
  }
}

// ===========================================================================
// SECTION D : Fused Adam optimizer.
// ===========================================================================

// ---------------------------------------------------------------------------
// Fused Adam update :
//   m  = b1*m + (1-b1)*g
//   v  = b2*v + (1-b2)*g^2
//   m_hat = m / (1 - b1^t)
//   v_hat = v / (1 - b2^t)
//   p -= lr * m_hat / (sqrt(v_hat) + eps)
// ---------------------------------------------------------------------------
__global__ void adamKernel(float* params, const float* grads,
                           float* m, float* v,
                           float lr, float beta1, float beta2,
                           float eps,
                           float bc1,            // 1 - beta1^t
                           float bc2,            // 1 - beta2^t
                           int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;
  float g  = grads[i];
  float mi = beta1 * m[i] + (1.0f - beta1) * g;
  float vi = beta2 * v[i] + (1.0f - beta2) * g * g;
  m[i] = mi;
  v[i] = vi;
  float m_hat = mi / bc1;
  float v_hat = vi / bc2;
  params[i] -= lr * m_hat / (sqrtf(v_hat) + eps);
}

// ===========================================================================
// Host driver / verification.
// ===========================================================================

static void cpuSigmoid (const float* x, float* y, int N) {
  for (int i = 0; i < N; ++i) y[i] = 1.0f / (1.0f + std::exp(-x[i]));
}
static void cpuTanh    (const float* x, float* y, int N) {
  for (int i = 0; i < N; ++i) y[i] = std::tanh(x[i]);
}
static void cpuLeakyRelu(const float* x, float* y, float a, int N) {
  for (int i = 0; i < N; ++i) y[i] = x[i] > 0 ? x[i] : a * x[i];
}
static void cpuReluBwd (const float* x, const float* dy, float* dx, int N) {
  for (int i = 0; i < N; ++i) dx[i] = x[i] > 0 ? dy[i] : 0.0f;
}

static float maxAbsErr(const float* a, const float* b, int N) {
  float e = 0.0f;
  for (int i = 0; i < N; ++i) {
    float d = std::fabs(a[i] - b[i]);
    if (d > e) e = d;
  }
  return e;
}

static void partA_activations() {
  printf("=== Part A : Activation kernels ===\n");
  const int N = 10'000'000;
  size_t bytes = N * sizeof(float);
  std::vector<float> hX(N), hRef(N), hOut(N);
  for (int i = 0; i < N; ++i) hX[i] = ((i % 1000) - 500) * 0.01f;

  float *dX, *dY;
  CUDA_CHECK(cudaMalloc(&dX, bytes));
  CUDA_CHECK(cudaMalloc(&dY, bytes));
  CUDA_CHECK(cudaMemcpy(dX, hX.data(), bytes, cudaMemcpyHostToDevice));

  int tpb = 256;
  int blk = (N + tpb - 1) / tpb;

  cudaEvent_t s, e;
  CUDA_CHECK(cudaEventCreate(&s));
  CUDA_CHECK(cudaEventCreate(&e));

  auto bw_gbs = [&](float ms){
    return (3.0f * bytes / 1.0e9f) / (ms / 1000.0f);   // r+w roughly = 2x; with bw use 3
  };

  // Sigmoid
  CUDA_CHECK(cudaEventRecord(s));
  for (int r = 0; r < 5; ++r) sigmoidFwd<<<blk, tpb>>>(dX, dY, N);
  CUDA_CHECK(cudaEventRecord(e));
  CUDA_CHECK(cudaEventSynchronize(e));
  float t_sig = 0.0f; CUDA_CHECK(cudaEventElapsedTime(&t_sig, s, e)); t_sig /= 5;
  CUDA_CHECK(cudaMemcpy(hOut.data(), dY, bytes, cudaMemcpyDeviceToHost));
  cpuSigmoid(hX.data(), hRef.data(), N);
  float err_sig = maxAbsErr(hOut.data(), hRef.data(), N);

  // Tanh
  CUDA_CHECK(cudaEventRecord(s));
  for (int r = 0; r < 5; ++r) tanhFwd<<<blk, tpb>>>(dX, dY, N);
  CUDA_CHECK(cudaEventRecord(e));
  CUDA_CHECK(cudaEventSynchronize(e));
  float t_tan = 0.0f; CUDA_CHECK(cudaEventElapsedTime(&t_tan, s, e)); t_tan /= 5;
  CUDA_CHECK(cudaMemcpy(hOut.data(), dY, bytes, cudaMemcpyDeviceToHost));
  cpuTanh(hX.data(), hRef.data(), N);
  float err_tan = maxAbsErr(hOut.data(), hRef.data(), N);

  // Leaky ReLU
  CUDA_CHECK(cudaEventRecord(s));
  for (int r = 0; r < 5; ++r) leakyReluFwd<<<blk, tpb>>>(dX, dY, 0.01f, N);
  CUDA_CHECK(cudaEventRecord(e));
  CUDA_CHECK(cudaEventSynchronize(e));
  float t_lr = 0.0f; CUDA_CHECK(cudaEventElapsedTime(&t_lr, s, e)); t_lr /= 5;
  CUDA_CHECK(cudaMemcpy(hOut.data(), dY, bytes, cudaMemcpyDeviceToHost));
  cpuLeakyRelu(hX.data(), hRef.data(), 0.01f, N);
  float err_lr = maxAbsErr(hOut.data(), hRef.data(), N);

  // ReLU backward
  std::vector<float> hDy(N, 0.5f);
  float *dDy, *dDx;
  CUDA_CHECK(cudaMalloc(&dDy, bytes));
  CUDA_CHECK(cudaMalloc(&dDx, bytes));
  CUDA_CHECK(cudaMemcpy(dDy, hDy.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaEventRecord(s));
  for (int r = 0; r < 5; ++r) reluBwd<<<blk, tpb>>>(dX, dDy, dDx, N);
  CUDA_CHECK(cudaEventRecord(e));
  CUDA_CHECK(cudaEventSynchronize(e));
  float t_rb = 0.0f; CUDA_CHECK(cudaEventElapsedTime(&t_rb, s, e)); t_rb /= 5;
  CUDA_CHECK(cudaMemcpy(hOut.data(), dDx, bytes, cudaMemcpyDeviceToHost));
  cpuReluBwd(hX.data(), hDy.data(), hRef.data(), N);
  float err_rb = maxAbsErr(hOut.data(), hRef.data(), N);

  printf("%-12s %-10s %-10s %-12s\n", "kernel", "time(ms)", "GB/s", "max|err|");
  printf("%-12s %-10.3f %-10.1f %-12.2e\n", "sigmoid",   t_sig, bw_gbs(t_sig), err_sig);
  printf("%-12s %-10.3f %-10.1f %-12.2e\n", "tanh",      t_tan, bw_gbs(t_tan), err_tan);
  printf("%-12s %-10.3f %-10.1f %-12.2e\n", "leakyRelu", t_lr,  bw_gbs(t_lr),  err_lr);
  printf("%-12s %-10.3f %-10.1f %-12.2e\n", "reluBwd",   t_rb,  bw_gbs(t_rb),  err_rb);
  printf("\n");

  CUDA_CHECK(cudaFree(dX));
  CUDA_CHECK(cudaFree(dY));
  CUDA_CHECK(cudaFree(dDy));
  CUDA_CHECK(cudaFree(dDx));
  CUDA_CHECK(cudaEventDestroy(s));
  CUDA_CHECK(cudaEventDestroy(e));
}

static void partB_losses() {
  printf("=== Part B : Loss functions ===\n");

  // ----- BCE -----
  const int N = 1024;
  std::vector<float> hLog(N), hLab(N), hLoss(N);
  for (int i = 0; i < N; ++i) {
    hLog[i] = ((i % 21) - 10) * 0.5f;
    hLab[i] = (i % 2);
  }
  float *dLog, *dLab, *dLoss;
  CUDA_CHECK(cudaMalloc(&dLog,  N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dLab,  N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dLoss, N * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(dLog, hLog.data(), N * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dLab, hLab.data(), N * sizeof(float), cudaMemcpyHostToDevice));
  bceLossKernel<<<(N + 255) / 256, 256>>>(dLog, dLab, dLoss, N);
  CUDA_CHECK(cudaMemcpy(hLoss.data(), dLoss, N * sizeof(float), cudaMemcpyDeviceToHost));

  double mean_kern = 0.0;
  for (int i = 0; i < N; ++i) mean_kern += hLoss[i];
  mean_kern /= N;

  double mean_ref = 0.0;
  for (int i = 0; i < N; ++i) {
    float z = hLog[i], y = hLab[i];
    float sp = std::log(1.0 + std::exp(-std::abs(z)));
    mean_ref += std::max(z, 0.0f) - z * y + sp;
  }
  mean_ref /= N;
  printf("BCE  mean kernel = %.6f, ref = %.6f, |err| = %.2e\n",
         mean_kern, mean_ref, std::fabs(mean_kern - mean_ref));
  CUDA_CHECK(cudaFree(dLog));
  CUDA_CHECK(cudaFree(dLab));
  CUDA_CHECK(cudaFree(dLoss));

  // ----- CE  + grad -----
  const int B = 256, C = 10;
  std::vector<float> hLogits(B * C);
  std::vector<int>   hLbl(B);
  for (int i = 0; i < B; ++i) {
    hLbl[i] = i % C;
    for (int c = 0; c < C; ++c)
      hLogits[i * C + c] = std::sin(0.1f * (i + c));
  }
  float *dL; int *dLB;
  CUDA_CHECK(cudaMalloc(&dL,  B * C * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dLB, B * sizeof(int)));
  CUDA_CHECK(cudaMemcpy(dL,  hLogits.data(), B * C * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dLB, hLbl.data(),    B * sizeof(int),       cudaMemcpyHostToDevice));

  float *dCE, *dG;
  CUDA_CHECK(cudaMalloc(&dCE, B * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dG,  B * C * sizeof(float)));
  crossEntropyKernel    <<<(B + 31) / 32, 32>>>(dL, dLB, dCE, B, C);
  crossEntropyGradKernel<<<(B + 31) / 32, 32>>>(dL, dLB, dG,  B, C);

  std::vector<float> hCE(B), hG(B * C);
  CUDA_CHECK(cudaMemcpy(hCE.data(), dCE, B * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(hG.data(),  dG,  B * C * sizeof(float), cudaMemcpyDeviceToHost));

  // CPU reference.
  std::vector<float> rCE(B), rG(B * C);
  for (int i = 0; i < B; ++i) {
    float maxv = hLogits[i*C];
    for (int c = 1; c < C; ++c) maxv = std::max(maxv, hLogits[i*C+c]);
    float sumE = 0.0f;
    for (int c = 0; c < C; ++c) sumE += std::exp(hLogits[i*C+c] - maxv);
    float lse = maxv + std::log(sumE);
    rCE[i] = -hLogits[i*C + hLbl[i]] + lse;
    float invN = 1.0f / B;
    for (int c = 0; c < C; ++c) {
      float p = std::exp(hLogits[i*C+c] - maxv) / sumE;
      rG[i*C+c] = (p - (c == hLbl[i] ? 1.0f : 0.0f)) * invN;
    }
  }
  float ce_err = maxAbsErr(hCE.data(), rCE.data(), B);
  float g_err  = maxAbsErr(hG.data(),  rG.data(),  B*C);
  printf("CE   max|err| = %.2e, dCE/dlogits max|err| = %.2e\n\n", ce_err, g_err);

  CUDA_CHECK(cudaFree(dL));
  CUDA_CHECK(cudaFree(dLB));
  CUDA_CHECK(cudaFree(dCE));
  CUDA_CHECK(cudaFree(dG));
}

static void partC_adam() {
  printf("=== Part C : Fused Adam (100 steps vs CPU reference) ===\n");
  const int N = 1024;
  std::vector<float> hP(N), hRefP(N);
  std::vector<float> hG(N);
  for (int i = 0; i < N; ++i) {
    hP[i]   = 0.0f;
    hRefP[i]= 0.0f;
    hG[i]   = 0.001f * ((i % 7) - 3);
  }

  float *dP, *dG, *dM, *dV;
  CUDA_CHECK(cudaMalloc(&dP, N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dG, N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dM, N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dV, N * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(dP, hP.data(), N * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dG, hG.data(), N * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(dM, 0, N * sizeof(float)));
  CUDA_CHECK(cudaMemset(dV, 0, N * sizeof(float)));

  std::vector<float> rM(N, 0.0f), rV(N, 0.0f);

  const float lr = 1e-3f, b1 = 0.9f, b2 = 0.999f, eps = 1e-8f;
  for (int t = 1; t <= 100; ++t) {
    float bc1 = 1.0f - std::pow(b1, t);
    float bc2 = 1.0f - std::pow(b2, t);
    adamKernel<<<(N+255)/256, 256>>>(dP, dG, dM, dV, lr, b1, b2, eps, bc1, bc2, N);
    // CPU reference.
    for (int i = 0; i < N; ++i) {
      rM[i] = b1 * rM[i] + (1 - b1) * hG[i];
      rV[i] = b2 * rV[i] + (1 - b2) * hG[i] * hG[i];
      float m_hat = rM[i] / bc1;
      float v_hat = rV[i] / bc2;
      hRefP[i] -= lr * m_hat / (std::sqrt(v_hat) + eps);
    }
  }

  std::vector<float> hOut(N);
  CUDA_CHECK(cudaMemcpy(hOut.data(), dP, N * sizeof(float), cudaMemcpyDeviceToHost));
  printf("Adam max|err| after 100 steps = %.2e\n\n", maxAbsErr(hOut.data(), hRefP.data(), N));

  CUDA_CHECK(cudaFree(dP));
  CUDA_CHECK(cudaFree(dG));
  CUDA_CHECK(cudaFree(dM));
  CUDA_CHECK(cudaFree(dV));
}

int main() {
  partA_activations();
  partB_losses();
  partC_adam();
  return 0;
}
