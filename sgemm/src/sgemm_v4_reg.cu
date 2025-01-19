#include "sgemm_v4_reg.cuh"

#define FETCH_FLOAT4(ptr) (reinterpret_cast<float4 *>(&(ptr))[0])
template <int M_NUM_PER_BLOCK, int N_NUM_PER_BLOCK, int K_NUM_PER_BLOCK, int NUM_PER_THREAD>
__global__ void sgemm4(float *A, float *B, float *C, const int M, const int N, const int K)
{
    constexpr int REG_NUM = NUM_PER_THREAD >> 1;
    int ctid[2];
    reIndex(ctid, N_NUM_PER_BLOCK / REG_NUM);
    int ctid_x = ctid[0];
    int ctid_y = ctid[1];

    int ltid[2];
    reIndex(ltid, N_NUM_PER_BLOCK / NUM_PER_THREAD);
    int ltid_x = ltid[0];
    int ltid_y = ltid[1];

    float* A_start = A + (blockIdx.y * M_NUM_PER_BLOCK) * K;
    float* B_start = B + (blockIdx.x * K_NUM_PER_BLOCK);

    __shared__ float shared_A[M_NUM_PER_BLOCK][K_NUM_PER_BLOCK];
    __shared__ float shared_B[K_NUM_PER_BLOCK][N_NUM_PER_BLOCK];

    float a_reg[REG_NUM] = {0.f};
    float b_reg[REG_NUM] = {0.f};
    float sum[REG_NUM][REG_NUM] = {0.f};

int inner_j = ltid_x * NUM_PER_THREAD;
#pragma unroll
    for (int s = 0; s < K; s += K_NUM_PER_BLOCK)
    {
        FETCH_FLOAT4(shared_A[ltid_y][inner_j]) = FETCH_FLOAT4(A_start[ltid_y * K + (s + inner_j)]);
        FETCH_FLOAT4(shared_B[ltid_y][inner_j]) = FETCH_FLOAT4(B_start[(ltid_y + s) * N + inner_j]);
        __syncthreads();

        for (int k = 0; k < K_NUM_PER_BLOCK; ++k)
        {
            a_reg[0] = shared_A[ctid_y * REG_NUM][k];
            a_reg[1] = shared_A[ctid_y * REG_NUM + 1][k];
            b_reg[0] = shared_B[k][ctid_x * REG_NUM];
            b_reg[1] = shared_B[k][ctid_x * REG_NUM + 1];
#pragma unroll
            for (int ii = 0; ii < REG_NUM; ii++)
            {
#pragma unroll
                for (int jj = 0; jj < REG_NUM; jj++)
                {
                    sum[ii][jj] += a_reg[ii] * b_reg[jj];
                }
            }
        }

        __syncthreads();
    }

    float *C_start = C + blockIdx.y * M_NUM_PER_BLOCK * N + blockIdx.x * N_NUM_PER_BLOCK;
#pragma unroll
    for (int i = 0; i < REG_NUM; i++)
    {
        for (int j = 0; j < REG_NUM; j++)
        {
            C_start[(ctid_y * REG_NUM + i) * N + (ctid_x * REG_NUM + j)] = sum[i][j];
        }
    }
}

template __global__ void sgemm4<32, 32, 32, 4>(float *A, float *B, float *C, const int M, const int N, const int K);

__device__ __forceinline__ void reIndex(int *new_index, int x_dim)
{
    new_index[0] = threadIdx.x % x_dim;
    new_index[1] = threadIdx.x / x_dim;
}