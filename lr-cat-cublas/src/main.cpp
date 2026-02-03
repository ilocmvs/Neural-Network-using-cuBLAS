#include <iostream>
#include <vector>
#include "dataset.h"
#include "lr_cublas.h"
#include "timer.h"

static float accuracy_cpu(const std::vector<float>& p, const std::vector<float>& y) {
  int correct = 0;
  for (size_t i = 0; i < y.size(); i++) {
    int pred = (p[i] >= 0.5f) ? 1 : 0;
    int gt   = (y[i] >= 0.5f) ? 1 : 0;
    correct += (pred == gt);
  }
  return (float)correct / (float)y.size();
}

int main(int argc, char** argv) {
  int N = 4096;
  int W = 32, H = 32;
  float lr = 0.1f;
  int iters = 200;

  if (argc > 1) N = std::atoi(argv[1]);
  if (argc > 2) iters = std::atoi(argv[2]);

  Dataset ds = load_dataset_stub("./data", N, W, H, true);

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  LrCublas model;
  model.init(ds.N, ds.D, stream);
  model.upload_Xy(ds.X.data(), ds.y.data());
  model.random_init(1234ULL);

  std::vector<float> h_p(ds.N);

  GpuTimer t;
  for (int it = 0; it < iters; it++) {
    t.tic(stream);
    float loss = model.train_step(lr);
    float ms = t.toc(stream);

    // Pull probabilities occasionally just to compute accuracy (for debugging)
    if (it % 20 == 0 || it == iters - 1) {
      CUDA_CHECK(cudaMemcpyAsync(h_p.data(), model.d_p, ds.N * sizeof(float),
                                cudaMemcpyDeviceToHost, stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));
      float acc = accuracy_cpu(h_p, ds.y);
      std::cout << "iter " << it
                << " loss=" << loss
                << " acc=" << acc
                << " step_ms=" << ms
                << "\n";
    }
  }

  model.cleanup();
  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}
