#pragma once
#include <cublas_v2.h>
#include <cuda_runtime.h>

struct LrCublas {
  int N = 0;
  int D = 0;

  // Device buffers
  float* d_X = nullptr;    // N*D
  float* d_y = nullptr;    // N
  float* d_W = nullptr;    // D
  float* d_z = nullptr;    // N
  float* d_p = nullptr;    // N
  float* d_e = nullptr;    // N   (p - y)
  float* d_dW = nullptr;   // D
  float* d_db = nullptr;   // 1

  float h_b = 0.0f;        // bias on host (simple)
  cublasHandle_t handle = nullptr;
  cudaStream_t stream = 0;

  void init(int N_, int D_, cudaStream_t s=0);
  void upload_Xy(const float* h_X, const float* h_y);
  void random_init(unsigned long long seed=1234ULL);

  // One training step on full batch
  // returns loss (host)
  float train_step(float lr);

  // Inference probability for batch X
  void predict_proba(float* d_out_p /*N*/, const float* d_in_X /*N*D*/);

  void cleanup();
};
