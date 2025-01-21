#pragma once

template <int M_NUM_PER_BLOCK,
          int N_NUM_PER_BLOCK, 
          int K_NUM_PER_BLOCK, 
          int M_NUM_PER_THREAD,
          int N_NUM_PER_THREAD,
          int K_NUM_PER_THREAD>
__global__ void sgemm6(float *A, float *B, float *C, const int M, const int N, const int K);

__device__ __forceinline__ void reIndex(int *new_index, int x_dim);