#pragma once
#include <vector>
#include <cstdint>
#include <string>

struct Dataset {
  int N = 0;      // number of samples
  int D = 0;      // flattened feature dim (e.g. 32*32*3)
  std::vector<float> X; // size N*D, row-major: X[i*D + j]
  std::vector<float> y; // size N, labels in {0,1}
};

// For the course: implement however you like:
// - CIFAR-10: cat vs not-cat
// - a folder of images with labels
Dataset load_dataset_stub(const std::string& root_dir,
                          int max_samples,
                          int target_w,
                          int target_h,
                          bool normalize_0_1);
