#pragma once

template<int BLOCKSIZE>
__global__ void sgemm1(float *A, float *B, float *C, const int M, const int N, const int K);