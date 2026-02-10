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
  float lr = 0.01f;
  int iters = 200;

  if (argc > 1) N = std::atoi(argv[1]);
  if (argc > 2) iters = std::atoi(argv[2]);
  if (argc > 3) lr = std::atof(argv[3]);

  Dataset ds = load_dataset_cifar10_cat("./data/cifar-10-batches-bin", N);

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
    if (it % 1000 == 0) {
      CUDA_CHECK(cudaMemcpyAsync(h_p.data(), model.d_p, ds.N * sizeof(float),
                                cudaMemcpyDeviceToHost, stream));
      //Debugging part of confusion matrix and metrics
      // CUDA_CHECK(cudaStreamSynchronize(stream));
      // int tp=0, tn=0, fp=0, fn=0;
      // int pos=0, neg=0;
      // double mean_p = 0.0;

      // for (int i = 0; i < ds.N; i++) {
      //   int gt = (ds.y[i] >= 0.5f) ? 1 : 0;
      //   int pr = (h_p[i] >= 0.5f) ? 1 : 0;
      //   mean_p += h_p[i];

      //   if (gt) pos++; else neg++;

      //   if (pr==1 && gt==1) tp++;
      //   else if (pr==0 && gt==0) tn++;
      //   else if (pr==1 && gt==0) fp++;
      //   else fn++;
      // }
      // mean_p /= ds.N;

      // double prec = (tp+fp) ? (double)tp/(tp+fp) : 0.0;
      // double rec  = (tp+fn) ? (double)tp/(tp+fn) : 0.0;
      // double tnr  = (tn+fp) ? (double)tn/(tn+fp) : 0.0; // specificity
      // double bal_acc = 0.5*(rec + tnr);

      // std::cout
      //   << "pos%=" << (double)pos/ds.N
      //   << " mean_p=" << mean_p
      //   << " TP=" << tp << " FP=" << fp << " TN=" << tn << " FN=" << fn
      //   << " bal_acc=" << bal_acc
      //   << " prec=" << prec
      //   << " rec=" << rec
      //   << "\n";

      CUDA_CHECK(cudaStreamSynchronize(stream));
      float acc = accuracy_cpu(h_p, ds.y);
      std::cout << "iter " << it
                << " loss=" << loss
                << " acc=" << acc
                << " step_ms=" << ms
                << "\n";
    }
  }
  
  // --------------------
  // Evaluate on test set
  // --------------------
  Dataset test_ds = load_dataset_cifar10_cat_test(
      "./data/cifar-10-batches-bin", 10000);

  // Allocate device buffer for test X and output probs
  float* d_X_test = nullptr;
  float* d_z_test = nullptr;
  float* d_p_test = nullptr;

  CUDA_CHECK(cudaMalloc(&d_X_test,
      (size_t)test_ds.N * test_ds.D * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_z_test,
      (size_t)test_ds.N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_p_test,
      (size_t)test_ds.N * sizeof(float)));
  

  CUDA_CHECK(cudaMemcpyAsync(
      d_X_test, test_ds.X.data(),
      (size_t)test_ds.N * test_ds.D * sizeof(float),
      cudaMemcpyHostToDevice, stream));

  // Inference
  model.predict_proba(d_p_test, d_z_test, d_X_test, test_ds.N);

  // Copy probabilities back
  std::vector<float> h_p_test(test_ds.N);
  CUDA_CHECK(cudaMemcpyAsync(
      h_p_test.data(), d_p_test,
      (size_t)test_ds.N * sizeof(float),
      cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  // Compute accuracy
  int correct = 0;
  for (int i = 0; i < test_ds.N; i++) {
    int pred = (h_p_test[i] >= 0.5f) ? 1 : 0;
    int gt   = (test_ds.y[i] >= 0.5f) ? 1 : 0;
    correct += (pred == gt);
  }

  float test_acc = (float)correct / (float)test_ds.N;
  std::cout << "Test accuracy: " << test_acc << "\n";

  // Cleanup test buffers
  cudaFree(d_X_test);
  cudaFree(d_p_test);

  model.cleanup();
  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}
