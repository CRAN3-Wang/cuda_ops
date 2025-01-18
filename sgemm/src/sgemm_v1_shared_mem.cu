#include "sgemm_v1_shared_mem.cuh"

template<int BLOCKSIZE>
__global__ void sgemm1(float *A, float *B, float *C, const int M, const int N, const int K)
{
    const int tid_x = threadIdx.x;
    const int tid_y = threadIdx.y;

    const int global_block_start_x = blockIdx.x * blockDim.x;
    const int global_block_start_y = blockIdx.y * blockDim.y;

    // Register per thread
    float sum = 0.f;
    for (size_t i = 0; i * BLOCKSIZE < K; i++)
    {
        __shared__ float shared_A[BLOCKSIZE][BLOCKSIZE];
        __shared__ float shared_B[BLOCKSIZE][BLOCKSIZE];

        size_t offsetA = i * BLOCKSIZE;
        size_t offsetB = i * BLOCKSIZE * N;
        
        int inner_idx_A = tid_y * K + tid_x;
        int inner_idx_B = tid_y * N + tid_x;

        shared_A[tid_y][tid_x] = A[global_block_start_y * K + offsetA + inner_idx_A];
        shared_B[tid_y][tid_x] = B[global_block_start_x + offsetB + inner_idx_B];

        __syncthreads();

#pragma unroll
        for (int i = 0; i < BLOCKSIZE; i++)
        {
            sum += shared_A[tid_y][i] * shared_B[i][tid_x];
        }

        __syncthreads();
    }

    C[(global_block_start_y + tid_y) * N + global_block_start_x + tid_x] = sum;
}

template __global__ void sgemm1<16>(float *A, float *B, float *C, const int M, const int N, const int K);