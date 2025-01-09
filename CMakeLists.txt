cmake_minimum_required(VERSION 3.22.1)
project(CUDA_OPS VERSION 0.1.0 LANGUAGES CUDA CXX C)
find_package(CUDAToolkit REQUIRED)
add_subdirectory(reduce)