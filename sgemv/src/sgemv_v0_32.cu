#include "sgemv_v0_32.cuh"

template <unsigned int WARPSIZE>
__device__ __forceinline__ float warpReduce(float sum)
{
    if (WARPSIZE >= 32)
        sum += __shfl_down_sync(0xffffffff, sum, 16);
    if (WARPSIZE >= 16)
        sum += __shfl_down_sync(0xffffffff, sum, 8);
    if (WARPSIZE >= 8)
        sum += __shfl_down_sync(0xffffffff, sum, 4);
    if (WARPSIZE >= 4)
        sum += __shfl_down_sync(0xffffffff, sum, 2);
    if (WARPSIZE >= 2)
        sum += __shfl_down_sync(0xffffffff, sum, 1);
    return sum;
}

// If n >= 32 && n < 128 ?
__global__ void sgemv0(float *__restrict__ A, float *__restrict__ x, float *__restrict__ y, const int M, const int N)
{
    const int tid_x = threadIdx.x;
    const int tid_y = threadIdx.y;

    const int WARPSIZE = 32;
    int lane_id = tid_x % WARPSIZE;
    int cur_row = blockDim.y * blockIdx.x + tid_y;

    if (cur_row < M)
    {
        float sum = 0.0f;
#pragma unroll
        for (int i = 0; i < N; i += WARPSIZE)
        {
            int cur_col = i + lane_id;
            sum += A[cur_row * N + cur_col] * x[cur_col];
        }
        sum = warpReduce<WARPSIZE>(sum);
        if (lane_id == 0)
            y[cur_row] = sum;
    }
}
