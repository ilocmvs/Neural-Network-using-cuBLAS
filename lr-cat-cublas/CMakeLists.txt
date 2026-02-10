cmake_minimum_required(VERSION 3.20)
project(lr_cat_cublas LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

find_package(CUDAToolkit REQUIRED)

add_executable(lr_cat_cublas
  src/main.cpp
  src/dataset_stub.cpp
  src/lr_cublas.cu
)

target_include_directories(lr_cat_cublas PRIVATE include)

target_link_libraries(lr_cat_cublas PRIVATE
  CUDA::cudart
  CUDA::cublas
)

# Optional: better debug
target_compile_options(lr_cat_cublas PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
)
