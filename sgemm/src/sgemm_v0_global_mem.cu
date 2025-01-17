#include "../include/sgemm_v0_global_mem.cuh"

__global__ void sgemm0(float *A, float *B, float *C, const int M, const int N, const int K)
{
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    int n = blockIdx.x * blockDim.x + threadIdx.x;

    if (m < M && n < N)
    {
        float temp = 0.f;
        for (int k = 0; k < K; ++k)
        {
            temp += A[m * K + k] * B[k * N + n];
        }
        C[m * N + n] = temp;
    }
}