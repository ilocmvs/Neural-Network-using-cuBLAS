#include "dataset.h"
#include <random>
#include <iostream>

// This stub generates a fake "cat vs not-cat" dataset.
// Replace with real image loading later (CIFAR-10 or folder).
Dataset load_dataset_stub(const std::string& root_dir,
                          int max_samples,
                          int target_w,
                          int target_h,
                          bool normalize_0_1) {
  (void)root_dir;
  (void)normalize_0_1;

  Dataset ds;
  ds.N = max_samples;
  ds.D = target_w * target_h * 3; // pretend RGB image flattened

  ds.X.resize((size_t)ds.N * ds.D);
  ds.y.resize(ds.N);

  std::mt19937 rng(42);
  std::normal_distribution<float> noise(0.f, 1.f);

  // Create a linearly-separable toy: “cats” have slightly higher mean on first K dims.
  int K = std::min(128, ds.D);
  for (int i = 0; i < ds.N; i++) {
    float label = (i % 2 == 0) ? 1.f : 0.f;
    ds.y[i] = label;
    for (int j = 0; j < ds.D; j++) {
      float v = noise(rng);
      if (j < K) v += (label > 0.5f) ? 0.75f : -0.75f;
      ds.X[(size_t)i * ds.D + j] = v;
    }
  }

  std::cerr << "[dataset_stub] Generated N=" << ds.N << " D=" << ds.D << "\n";
  return ds;
}
