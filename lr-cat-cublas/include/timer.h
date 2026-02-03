#pragma once
#include <cuda_runtime.h>
#include "cuda_check.h"

struct GpuTimer {
  cudaEvent_t start{}, stop{};
  GpuTimer() {
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
  }
  ~GpuTimer() {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  }
  void tic(cudaStream_t s=0) { CUDA_CHECK(cudaEventRecord(start, s)); }
  float toc(cudaStream_t s=0) {
    CUDA_CHECK(cudaEventRecord(stop, s));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    return ms;
  }
};
