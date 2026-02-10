#include "dataset.h"
#include <fstream>
#include <iostream>

namespace {//unanimous namespace for internal linkage
  
  void normalize_dataset(Dataset& ds) {
    int D = ds.D, N = ds.N;
    // Normalize features to zero mean and unit std (per dimension)
    std::vector<float> mean(D, 0.f), std(D, 0.f);
    // mean
    for (int i = 0; i < N; ++i)
      for (int j = 0; j < D; ++j)
        mean[j] += ds.X[i*D + j];
    for (int j = 0; j < D; ++j)
      mean[j] /= N;
    // std
    for (int i = 0; i < N; ++i)
      for (int j = 0; j < D; ++j) {
        float d = ds.X[i*D + j] - mean[j];
        std[j] += d * d;
      }
    for (int j = 0; j < D; ++j)
      std[j] = sqrt(std[j] / N) + 1e-6f;

    // normalize
    for (int i = 0; i < N; ++i)
      for (int j = 0; j < D; ++j)
        ds.X[i*D + j] = (ds.X[i*D + j] - mean[j]) / std[j];
    }
}

Dataset load_dataset_cifar10_cat(const std::string& root,
                                 int max_samples) {
  const int W = 32, H = 32, C = 3;
  const int D = W * H * C;
  const int RECORD_SIZE = 1 + D;

  Dataset ds;
  ds.D = D;

  std::vector<std::string> files = {
    root + "/data_batch_1.bin",
    root + "/data_batch_2.bin",
    root + "/data_batch_3.bin",
    root + "/data_batch_4.bin",
    root + "/data_batch_5.bin"
  };

  for (const auto& path : files) {
    std::ifstream fin(path, std::ios::binary);
    if (!fin) {
      std::cerr << "Failed to open " << path << "\n";
      continue;
    }

    while (fin && (int)ds.y.size() < max_samples) {
      unsigned char buffer[RECORD_SIZE];
      fin.read((char*)buffer, RECORD_SIZE);
      if (!fin) break;

      unsigned char label = buffer[0];

      // Binary label: cat vs not-cat
      ds.y.push_back(label == 3 ? 1.0f : 0.0f);

      // Flattened image: R then G then B
      size_t base = ds.X.size();
      ds.X.resize(base + D);

      for (int i = 0; i < D; i++) {
        ds.X[base + i] = buffer[1 + i];
      }

      
    }
  }
  ds.N = (int)ds.y.size();
  normalize_dataset(ds);

  std::cerr << "[CIFAR-10] Loaded " << ds.N << " samples\n";
  return ds;
}

Dataset load_dataset_cifar10_cat_test(const std::string& root,
                                      int max_samples) {
  const int W = 32, H = 32, C = 3;
  const int D = W * H * C;
  const int RECORD_SIZE = 1 + D;

  Dataset ds;
  ds.D = D;

  std::string path = root + "/test_batch.bin";
  std::ifstream fin(path, std::ios::binary);
  if (!fin) {
    std::cerr << "Failed to open " << path << "\n";
    return ds;
  }

  while (fin && (int)ds.y.size() < max_samples) {
    unsigned char buffer[RECORD_SIZE];
    fin.read((char*)buffer, RECORD_SIZE);
    if (!fin) break;

    unsigned char label = buffer[0];
    ds.y.push_back(label == 3 ? 1.0f : 0.0f);

    size_t base = ds.X.size();
    ds.X.resize(base + D);
    for (int i = 0; i < D; i++)
      ds.X[base + i] = buffer[1 + i] / 255.0f;
  }

  ds.N = (int)ds.y.size();
  normalize_dataset(ds);
  std::cerr << "[CIFAR-10 test] Loaded " << ds.N << " samples\n";
  return ds;
}