#include "sgemm_v5_reg_float4.cuh"

#define FETCH_FLOAT4(ptr) (reinterpret_cast<float4 *>(&(ptr))[0])
template <int M_NUM_PER_BLOCK,
          int N_NUM_PER_BLOCK,
          int K_NUM_PER_BLOCK,
          int M_NUM_PER_THREAD,
          int N_NUM_PER_THREAD,
          int K_NUM_PER_THREAD>
__global__ void sgemm5(float *A, float *B, float *C, const int M, const int N, const int K)
{
    int tid[2];
    reIndex(tid, N_NUM_PER_BLOCK / N_NUM_PER_THREAD);
    int tid_x = tid[0];
    int tid_y = tid[1];

    float *A_start = A + (blockIdx.y * M_NUM_PER_BLOCK) * K;
    float *B_start = B + (blockIdx.x * K_NUM_PER_BLOCK);

    __shared__ float shared_A[M_NUM_PER_BLOCK][K_NUM_PER_BLOCK];
    __shared__ float shared_B[K_NUM_PER_BLOCK][N_NUM_PER_BLOCK];

    float a_reg[M_NUM_PER_THREAD] = {0.f};
    float b_reg[N_NUM_PER_THREAD] = {0.f};
    float sum[M_NUM_PER_THREAD][N_NUM_PER_THREAD] = {0.f};

    for (int s = 0; s < K; s += K_NUM_PER_BLOCK)
    {
#pragma unroll
        for (int i = 0; i < M_NUM_PER_THREAD; ++i)
        {
            FETCH_FLOAT4(shared_A[tid_y * M_NUM_PER_THREAD + i][tid_x * K_NUM_PER_THREAD]) =
                FETCH_FLOAT4(A_start[(tid_y * M_NUM_PER_THREAD + i) * K + (s + (tid_x * K_NUM_PER_THREAD))]);
        }
#pragma unroll
        for (int i = 0; i < K_NUM_PER_THREAD; ++i)
        {
            FETCH_FLOAT4(shared_B[tid_y * K_NUM_PER_THREAD + i][tid_x * N_NUM_PER_THREAD]) =
                FETCH_FLOAT4(B_start[((tid_y * K_NUM_PER_THREAD + i) + s) * N + (tid_x * N_NUM_PER_THREAD)]);
        }
        __syncthreads();

        for (int k = 0; k < K_NUM_PER_BLOCK; ++k)
        {
            a_reg[0] = shared_A[tid_y * M_NUM_PER_THREAD][k];
            a_reg[1] = shared_A[tid_y * M_NUM_PER_THREAD + 1][k];
            a_reg[2] = shared_A[tid_y * M_NUM_PER_THREAD + 2][k];
            a_reg[3] = shared_A[tid_y * M_NUM_PER_THREAD + 3][k];
            FETCH_FLOAT4(b_reg[0]) = FETCH_FLOAT4(shared_B[k][tid_x * K_NUM_PER_THREAD]);
            
#pragma unroll
            for (int ii = 0; ii < M_NUM_PER_THREAD; ii++)
            {
#pragma unroll
                for (int jj = 0; jj < N_NUM_PER_THREAD; jj++)
                {
                    sum[ii][jj] += a_reg[ii] * b_reg[jj];
                }
            }
        }

        __syncthreads();
    }

    float *C_start = C + blockIdx.y * M_NUM_PER_BLOCK * N + blockIdx.x * N_NUM_PER_BLOCK;
#pragma unroll
    for (int i = 0; i < M_NUM_PER_THREAD; i++)
    {
        for (int j = 0; j < N_NUM_PER_THREAD; j++)
        {
            C_start[(tid_y * M_NUM_PER_THREAD + i) * N + (tid_x * N_NUM_PER_THREAD + j)] = sum[i][j];
        }
    }
}

template __global__ void sgemm5<64, 64, 64, 4, 4, 4>(float *A, float *B, float *C, const int M, const int N, const int K);

__device__ __forceinline__ void reIndex(int *new_index, int x_dim)
{
    new_index[0] = threadIdx.x % x_dim;
    new_index[1] = threadIdx.x / x_dim;
}