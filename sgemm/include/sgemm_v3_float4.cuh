#pragma once

template <int M_NUM_PER_BLOCK, int N_NUM_PER_BLOCK, int K_NUM_PER_BLOCK, int NUM_PER_THREAD>
__global__ void sgemm3(float *A, float *B, float *C, const int M, const int N, const int K);