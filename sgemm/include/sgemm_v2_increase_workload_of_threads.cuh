#pragma once

template<int BLOCKSIZE, int STRIDE>
__global__ void sgemm2(float *A, float *B, float *C, const int M, const int N, const int K);