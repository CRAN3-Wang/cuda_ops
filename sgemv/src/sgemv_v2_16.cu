#include "sgemv_v2_16.cuh"

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

// If n <= 16
template <unsigned int ROW_PER_WARP>
__global__ void sgemv2(float *__restrict__ A, float *__restrict__ x, float *__restrict__ y, const int M, const int N)
{
    const int tid_x = threadIdx.x;
    const int tid_y = threadIdx.y;

    const int WARPSIZE = 32;
    int lane_id = tid_x % WARPSIZE;
    const int THREAD_PER_ROW = WARPSIZE / ROW_PER_WARP;
    int cur_row = (blockDim.y * blockIdx.x + tid_y) * ROW_PER_WARP + lane_id / THREAD_PER_ROW;

    if (cur_row < M)
    {
        float sum = 0.0f;
        int cur_col = lane_id % THREAD_PER_ROW;
        sum += A[cur_row * N + cur_col] * x[cur_col];
        sum = warpReduce<THREAD_PER_ROW>(sum);
        if (cur_col == 0)
            y[cur_row] = sum;
    }
}

template __global__ void sgemv2<2>(float *__restrict__ A, float *__restrict__ x, float *__restrict__ y, const int M, const int N);