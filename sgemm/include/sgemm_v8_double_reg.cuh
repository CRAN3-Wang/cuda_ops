#pragma once

template <const int M_NUM_PER_BLOCK,
          const int N_NUM_PER_BLOCK,
          const int K_NUM_PER_BLOCK,
          const int Y_NUM_PER_THREAD,
          const int X_NUM_PER_THREAD>
__global__ void sgemm8(float *A, float *B, float *C, const int M, const int N, const int K);

__device__ __forceinline__ void reIndex(int *new_index, int x_dim);