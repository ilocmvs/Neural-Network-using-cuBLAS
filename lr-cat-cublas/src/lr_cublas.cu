#include "lr_cublas.h"
#include "cuda_check.h"
#include <cmath>
#include <cstdio>
#include <vector>

// -------------------------
// CUDA kernels (algorithm parts)
// -------------------------

__global__ void k_set_zero(float* a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) a[i] = 0.0f;
}

__global__ void k_add_bias(float* z, float b, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) z[i] += b;
}

__global__ void k_sigmoid(float* p, const float* z, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float x = z[i];
    // stable-ish sigmoid
    if (x >= 0) {
      float e = expf(-x);
      p[i] = 1.0f / (1.0f + e);
    } else {
      float e = expf(x);
      p[i] = e / (1.0f + e);
    }
  }
}

__global__ void k_error(float* e, const float* p, const float* y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (i < n) e[i] = (p[i] - y[i]) * (y[i] > 0.5f ? 9.0f : 1.0f); // for stable gradient when y=0 or 1
}

// Binary cross entropy loss: average over N
// loss = -mean( y*log(p) + (1-y)*log(1-p) )
__global__ void k_bce_logits_partial(const float* z, const float* y,
                                     float* partial, int n) {
  extern __shared__ float buf[];
  int tid = threadIdx.x;
  int i = blockIdx.x * blockDim.x + tid;

  float s = 0.f;
  if (i < n) {
    float zi = z[i];
    float yi = y[i];          // assume 0 or 1

    // Stable logistic loss:
    // L = max(z,0) - z*y + log(1 + exp(-|z|))
    float az = fabsf(zi);
    s = fmaxf(zi, 0.0f) - zi * yi + log1pf(expf(-az));
  }

  buf[tid] = s;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) buf[tid] += buf[tid + stride];
    __syncthreads();
  }
  if (tid == 0) partial[blockIdx.x] = buf[0];
}


__global__ void k_reduce_sum(const float* a, float* out, int n) {
  extern __shared__ float buf[];
  int tid = threadIdx.x;
  int i = blockIdx.x * blockDim.x + tid;
  float s = (i < n) ? a[i] : 0.f;
  buf[tid] = s;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) buf[tid] += buf[tid + stride];
    __syncthreads();
  }
  if (tid == 0) atomicAdd(out, buf[0]);
}

__global__ void k_sgd_update(float* w, const float* dw, float lr, float scale, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) w[i] -= lr * (dw[i] * scale);
}

// -------------------------
// Helper: init random W (simple LCG on GPU)
// -------------------------
__global__ void k_init_weights(float* w, int n, unsigned long long seed) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    unsigned long long x = seed ^ (0x9E3779B97F4A7C15ULL + (unsigned long long)i);
    // xorshift*
    x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
    float u = (float)((x * 2685821657736338717ULL) & 0x00FFFFFF) / (float)0x01000000;
    w[i] = (u - 0.5f) * 0.01f;
  }
}

// -------------------------
// LrCublas methods
// -------------------------

void LrCublas::init(int N_, int D_, cudaStream_t s) {
  N = N_;
  D = D_;
  stream = s;

  // Allocate device memory
  CUDA_CHECK(cudaMalloc(&d_X,  (size_t)N * D * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_y,  (size_t)N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_W,  (size_t)D * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_z,  (size_t)N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_p,  (size_t)N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_e,  (size_t)N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_dW, (size_t)D * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_db, sizeof(float)));

  // cuBLAS handle
  CUBLAS_CHECK(cublasCreate(&handle));
  CUBLAS_CHECK(cublasSetStream(handle, stream));
}

void LrCublas::upload_Xy(const float* h_X, const float* h_y) {
  CUDA_CHECK(cudaMemcpyAsync(d_X, h_X, (size_t)N * D * sizeof(float),
                            cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(d_y, h_y, (size_t)N * sizeof(float),
                            cudaMemcpyHostToDevice, stream));
}

void LrCublas::random_init(unsigned long long seed) {
  int threads = 256;
  int blocks = (D + threads - 1) / threads;
  k_init_weights<<<blocks, threads, 0, stream>>>(d_W, D, seed);
  CUDA_CHECK(cudaGetLastError());
  h_b = 0.0f;
}

void LrCublas::predict_proba(float* d_out_p, float* d_test_z, const float* d_in_X, int test_N) {
  float alpha = 1.0f, beta = 0.0f;
  CUBLAS_CHECK(cublasSgemv(handle, CUBLAS_OP_T, D, test_N, &alpha, d_in_X, D, d_W, 1, &beta, d_test_z, 1));
  // After GEMM, add bias and sigmoid:
  int threads = 256;
  int blocksN = (test_N + threads - 1) / threads;
  k_add_bias<<<blocksN, threads, 0, stream>>>(d_test_z, h_b, test_N);
  k_sigmoid<<<blocksN, threads, 0, stream>>>(d_out_p, d_test_z, test_N);
  CUDA_CHECK(cudaGetLastError());
}

float LrCublas::train_step(float lr) {
  // Forward: z = XW + b, p = sigmoid(z)
  // Backward: e = p - y
  //           dW = (1/N) X^T e
  //           db = (1/N) sum(e)
  // Update:   W -= lr*dW; b -= lr*db

  int threads = 256;
  int blocksN = (N + threads - 1) / threads;
  int blocksD = (D + threads - 1) / threads;

  // 0) clear db accumulator
  k_set_zero<<<1, 1, 0, stream>>>(d_db, 1);

  // 1) z = XW  (cuBLAS)
  float alpha = 1.0f, beta = 0.0f;
  CUBLAS_CHECK(cublasSgemv(handle, CUBLAS_OP_T, D, N, &alpha, d_X, D, d_W, 1, &beta, d_z, 1));
  CUDA_CHECK(cudaGetLastError());

  // 2) z += b
  k_add_bias<<<blocksN, threads, 0, stream>>>(d_z, h_b, N);

  // 3) p = sigmoid(z)
  k_sigmoid<<<blocksN, threads, 0, stream>>>(d_p, d_z, N);

  // 4) e = p - y
  k_error<<<blocksN, threads, 0, stream>>>(d_e, d_p, d_y, N);

  // 5) dW = X^T e  (cuBLAS)
  CUBLAS_CHECK(cublasSgemv(handle, CUBLAS_OP_N, D, N, &alpha, d_X, D, d_e, 1, &beta, d_dW, 1));

  // 6) db = sum(e) (reduction)
  // d_db already zeroed; atomicAdd inside.
  k_reduce_sum<<<blocksN, threads, threads * sizeof(float), stream>>>(d_e, d_db, N);

  // 7) Update weights: W -= lr * (1/N) * dW
  float invN = 1.0f / (float)N;
  k_sgd_update<<<blocksD, threads, 0, stream>>>(d_W, d_dW, lr, invN, D);

  // 8) Update bias on host (simple): b -= lr * (1/N) * db
  float h_db = 0.f;
  CUDA_CHECK(cudaMemcpyAsync(&h_db, d_db, sizeof(float),
                            cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  h_b -= lr * (h_db * invN);

  // 9) Compute loss (optional but useful)
  // We’ll do a simple 2-kernel reduction: partial sums then sum on CPU.
  int lossBlocks = (N + threads - 1) / threads;
  float* d_partial = nullptr;
  CUDA_CHECK(cudaMalloc(&d_partial, (size_t)lossBlocks * sizeof(float)));
  k_bce_logits_partial<<<lossBlocks, threads, threads * sizeof(float), stream>>>(d_z, d_y, d_partial, N);
  CUDA_CHECK(cudaGetLastError());

  std::vector<float> h_partial(lossBlocks);
  CUDA_CHECK(cudaMemcpyAsync(h_partial.data(), d_partial,
                            (size_t)lossBlocks * sizeof(float),
                            cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUDA_CHECK(cudaFree(d_partial));

  double sum = 0.0;
  for (float v : h_partial) sum += (double)v;
  float loss = (float)(sum / (double)N);

  return loss;
}

void LrCublas::cleanup() {
  if (d_X)  cudaFree(d_X);
  if (d_y)  cudaFree(d_y);
  if (d_W)  cudaFree(d_W);
  if (d_z)  cudaFree(d_z);
  if (d_p)  cudaFree(d_p);
  if (d_e)  cudaFree(d_e);
  if (d_dW) cudaFree(d_dW);
  if (d_db) cudaFree(d_db);

  d_X = d_y = d_W = d_z = d_p = d_e = d_dW = d_db = nullptr;

  if (handle) {
    cublasDestroy(handle);
    handle = nullptr;
  }
}
