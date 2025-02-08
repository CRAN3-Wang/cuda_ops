#pragma once

template <unsigned int WARPSIZE>
__device__ __forceinline__ float warpReduce(float sum);

__global__ void sgemv1(float *__restrict__ A, float *__restrict__ x, float *__restrict__ y, const int M, const int N);