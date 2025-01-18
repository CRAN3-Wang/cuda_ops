#include "sgemm_v3_float4.cuh"

#define inner_j (tid_x * NUM_PER_THREAD)
#define FETCH_FLOAT4(ptr) (reinterpret_cast<float4 *>(&(ptr))[0])
template <int M_NUM_PER_BLOCK, int N_NUM_PER_BLOCK, int K_NUM_PER_BLOCK, int NUM_PER_THREAD>
__global__ void sgemm3(float *A, float *B, float *C, const int M, const int N, const int K)
{
    int tid_x = threadIdx.x;
    int tid_y = threadIdx.y;

    int block_start_x = blockIdx.x * K_NUM_PER_BLOCK;
    int block_start_y = blockIdx.y * M_NUM_PER_BLOCK;

    __shared__ float shared_A[M_NUM_PER_BLOCK][K_NUM_PER_BLOCK];
    __shared__ float shared_B[K_NUM_PER_BLOCK][N_NUM_PER_BLOCK];

    float sum[NUM_PER_THREAD] = {0.f};

#pragma unroll
    for (int s = 0; s < K; s += K_NUM_PER_BLOCK)
    {
        FETCH_FLOAT4(shared_A[tid_y][inner_j]) = FETCH_FLOAT4(A[(block_start_y + tid_y) * K + s + inner_j]);
        FETCH_FLOAT4(shared_B[tid_y][inner_j]) = FETCH_FLOAT4(B[(tid_y + s) * N + block_start_x + inner_j]);
        __syncthreads();

#pragma unroll
        for (int i = 0; i < NUM_PER_THREAD; ++i)
        {
            for (int k = 0; k < K_NUM_PER_BLOCK; k++)
            {
                sum[i] += shared_A[tid_y][k] * shared_B[k][inner_j + i];
            }
        }
        __syncthreads();
    }

    float *C_start = C + blockIdx.y * M_NUM_PER_BLOCK * N + blockIdx.x * N_NUM_PER_BLOCK;
#pragma unroll
    for (int i = 0; i < NUM_PER_THREAD; ++i)
    {
        C_start[tid_y * N + tid_x * NUM_PER_THREAD + i] = sum[i];
    }
}

template __global__ void sgemm3<32, 32, 32, 4>(float *A, float *B, float *C, const int M, const int N, const int K);