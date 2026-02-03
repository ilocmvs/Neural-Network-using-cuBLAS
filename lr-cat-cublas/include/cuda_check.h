#pragma once
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CUDA_CHECK(call) do {                                      \
  cudaError_t err = (call);                                        \
  if (err != cudaSuccess) {                                        \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,   \
            cudaGetErrorString(err));                              \
    std::exit(EXIT_FAILURE);                                       \
  }                                                                \
} while(0)

static inline const char* cublasGetErrorString_(cublasStatus_t s) {
  switch (s) {
    case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
    case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
    default: return "CUBLAS_STATUS_UNKNOWN_ERROR";
  }
}

#define CUBLAS_CHECK(call) do {                                      \
  cublasStatus_t st = (call);                                        \
  if (st != CUBLAS_STATUS_SUCCESS) {                                 \
    fprintf(stderr, "cuBLAS error %s:%d: %s\n", __FILE__, __LINE__,   \
            cublasGetErrorString_(st));                              \
    std::exit(EXIT_FAILURE);                                         \
  }                                                                  \
} while(0)
