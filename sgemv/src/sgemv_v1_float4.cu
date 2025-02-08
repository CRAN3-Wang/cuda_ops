#include "sgemv_v1_float4.cuh"

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

// If n >= 128 ?
#define FETCH_FLOAT4(ptr) (reinterpret_cast<float4 *>(&(ptr))[0])
__global__ void sgemv1(float *__restrict__ A, float *__restrict__ x, float *__restrict__ y, const int M, const int N)
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
        for (int i = 0; i < N; i += 4 * WARPSIZE)
        {
            int cur_col = i + lane_id * 4;
            float4 cur_col_vec = FETCH_FLOAT4(x[cur_col]);
            float4 cur_row_vec = FETCH_FLOAT4(A[cur_row * N + cur_col]);
            sum += cur_col_vec.x * cur_row_vec.x;
            sum += cur_col_vec.y * cur_row_vec.y;
            sum += cur_col_vec.z * cur_row_vec.z;
            sum += cur_col_vec.w * cur_row_vec.w;
        }
        sum = warpReduce<WARPSIZE>(sum);
        if (lane_id == 0)
            y[cur_row] = sum;
    }
}