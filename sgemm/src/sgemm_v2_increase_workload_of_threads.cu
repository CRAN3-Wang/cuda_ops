#include "sgemm_v2_increase_workload_of_threads.cuh"

#define inner_i (ii * BLOCKSIZE + tid_y)
#define inner_j (jj * BLOCKSIZE + tid_x)

template <int BLOCKSIZE, int STRIDE>
__global__ void sgemm2(float *A, float *B, float *C, const int M, const int N, const int K)
{
    constexpr int STEP = BLOCKSIZE * STRIDE;

    const int tid_x = threadIdx.x;
    const int tid_y = threadIdx.y;

    const int global_block_start_x = blockIdx.x * STEP;
    const int global_block_start_y = blockIdx.y * STEP;

    // Register per thread
    float sum[STRIDE][STRIDE] = {0.f};
    for (size_t i = 0; i * STEP < K; i++)
    {
        __shared__ float shared_A[STEP][STEP];
        __shared__ float shared_B[STEP][STEP];

        size_t offsetA = i * STEP;
        size_t offsetB = i * STEP * N;

#pragma unroll
        for (int ii = 0; ii < STRIDE; ii++)
        {
#pragma unroll
            for (int jj = 0; jj < STRIDE; jj++)
            {
                shared_A[inner_i][inner_j] = A[global_block_start_y * K + offsetA + inner_i * K + inner_j];
                shared_B[inner_i][inner_j] = B[global_block_start_x + offsetB + inner_i * N + inner_j];
            }
        }
        __syncthreads();

#pragma unroll
        for (int ii = 0; ii < STRIDE; ii++)
        {
#pragma unroll
            for (int jj = 0; jj < STRIDE; jj++)
            {
#pragma unroll
                for (int kk = 0; kk < STEP; kk++)
                {

                    sum[ii][jj] += shared_A[inner_i][kk] * shared_B[kk][inner_j];
                }
            }
        }
        __syncthreads();
    }

    for (int ii = 0; ii < STRIDE; ii++)
    {
        for (int jj = 0; jj < STRIDE; jj++)
        {

            C[(global_block_start_y + inner_i) * N + global_block_start_x + inner_j] = sum[ii][jj];
        }
    }
}

template __global__ void sgemm2<16, 2>(float *A, float *B, float *C, const int M, const int N, const int K);