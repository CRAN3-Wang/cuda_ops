#include "sgemm_v7_double_buffer.cuh"

#define FETCH_FLOAT4(ptr) (reinterpret_cast<float4 *>(&(ptr))[0])
template <int M_NUM_PER_BLOCK,
          int N_NUM_PER_BLOCK,
          int K_NUM_PER_BLOCK,
          int X_NUM_PER_THREAD,
          int Y_NUM_PER_THREAD>
__global__ void sgemm7(float *A, float *B, float *C, const int M, const int N, const int K)
{
    // We have different layout for A and B, so we need two sets of tid for loading
    int atid[2];
    // Divide by 4 as 1 thread could load 4 floats using float4
    reIndex(atid, K_NUM_PER_BLOCK >> 2);
    int atid_x = atid[0];
    int atid_y = atid[1];

    int btid[2];
    reIndex(btid, N_NUM_PER_BLOCK >> 2);
    int btid_x = btid[0];
    int btid_y = btid[1];

    // And another set of tid for computing/C
    int ctid[2];
    reIndex(ctid, N_NUM_PER_BLOCK / X_NUM_PER_THREAD);
    int ctid_x = ctid[0];
    int ctid_y = ctid[1];

    float *A_start = A + (blockIdx.y * M_NUM_PER_BLOCK) * K;
    float *B_start = B + (blockIdx.x * N_NUM_PER_BLOCK);

    // Here we need two shared matrices for ping-pong buffering
    __shared__ float shared_A[2][K_NUM_PER_BLOCK][M_NUM_PER_BLOCK];
    __shared__ float shared_B[2][K_NUM_PER_BLOCK][N_NUM_PER_BLOCK];

    float a_reg[Y_NUM_PER_THREAD] = {0.f};
    float b_reg[X_NUM_PER_THREAD] = {0.f};
    float a_l_reg[4] = {0.f};
    float sum[Y_NUM_PER_THREAD][X_NUM_PER_THREAD] = {0.f};

    // Initial load and sync for the first stage (ping)
    FETCH_FLOAT4(a_l_reg[0]) = FETCH_FLOAT4(A_start[(atid_y * K) + (atid_x * 4)]);
    shared_A[0][atid_x * 4 + 0][atid_y] = a_l_reg[0];
    shared_A[0][atid_x * 4 + 1][atid_y] = a_l_reg[1];
    shared_A[0][atid_x * 4 + 2][atid_y] = a_l_reg[2];
    shared_A[0][atid_x * 4 + 3][atid_y] = a_l_reg[3];
    FETCH_FLOAT4(shared_B[0][btid_y][btid_x * 4]) = FETCH_FLOAT4(B_start[btid_y * N + btid_x * 4]);
    __syncthreads();

    int write_stage_idx = 1;
    for (int s = K_NUM_PER_BLOCK; s < K; s += K_NUM_PER_BLOCK)
    {
        // Load next stage (pong)
        FETCH_FLOAT4(a_l_reg[0]) = FETCH_FLOAT4(A_start[(atid_y * K) + s + (atid_x * 4)]);
        shared_A[write_stage_idx][atid_x * 4 + 0][atid_y] = a_l_reg[0];
        shared_A[write_stage_idx][atid_x * 4 + 1][atid_y] = a_l_reg[1];
        shared_A[write_stage_idx][atid_x * 4 + 2][atid_y] = a_l_reg[2];
        shared_A[write_stage_idx][atid_x * 4 + 3][atid_y] = a_l_reg[3];
        FETCH_FLOAT4(shared_B[write_stage_idx][btid_y][btid_x * 4]) = FETCH_FLOAT4(B_start[(btid_y + s) * N + (btid_x * 4)]);

        write_stage_idx ^= 1;
        for (int k = 0; k < K_NUM_PER_BLOCK; ++k)
        {
            FETCH_FLOAT4(a_reg[0]) = FETCH_FLOAT4(shared_A[write_stage_idx][k][ctid_y * Y_NUM_PER_THREAD + 0]);
            FETCH_FLOAT4(a_reg[4]) = FETCH_FLOAT4(shared_A[write_stage_idx][k][ctid_y * Y_NUM_PER_THREAD + 4]);
            FETCH_FLOAT4(b_reg[0]) = FETCH_FLOAT4(shared_B[write_stage_idx][k][ctid_x * X_NUM_PER_THREAD + 0]);
            FETCH_FLOAT4(b_reg[4]) = FETCH_FLOAT4(shared_B[write_stage_idx][k][ctid_x * X_NUM_PER_THREAD + 4]);

            // Unroll the loops to improve performance
#pragma unroll
            for (int ii = 0; ii < Y_NUM_PER_THREAD; ii++)
            {
#pragma unroll
                for (int jj = 0; jj < X_NUM_PER_THREAD; jj++)
                {
                    sum[ii][jj] += a_reg[ii] * b_reg[jj];
                }
            }
        }
        __syncthreads();
    }
    // Process the last stage
    write_stage_idx ^= 1;
    for (int k = 0; k < K_NUM_PER_BLOCK; ++k)
    {
        FETCH_FLOAT4(a_reg[0]) = FETCH_FLOAT4(shared_A[write_stage_idx][k][ctid_y * Y_NUM_PER_THREAD + 0]);
        FETCH_FLOAT4(a_reg[4]) = FETCH_FLOAT4(shared_A[write_stage_idx][k][ctid_y * Y_NUM_PER_THREAD + 4]);
        FETCH_FLOAT4(b_reg[0]) = FETCH_FLOAT4(shared_B[write_stage_idx][k][ctid_x * X_NUM_PER_THREAD + 0]);
        FETCH_FLOAT4(b_reg[4]) = FETCH_FLOAT4(shared_B[write_stage_idx][k][ctid_x * X_NUM_PER_THREAD + 4]);

#pragma unroll
        for (int ii = 0; ii < Y_NUM_PER_THREAD; ii++)
        {
#pragma unroll
            for (int jj = 0; jj < X_NUM_PER_THREAD; jj++)
            {
                sum[ii][jj] += a_reg[ii] * b_reg[jj];
            }
        }
    }

    float *C_start = C + blockIdx.y * M_NUM_PER_BLOCK * N + blockIdx.x * N_NUM_PER_BLOCK;
#pragma unroll
    for (int i = 0; i < Y_NUM_PER_THREAD; i++)
    {
        FETCH_FLOAT4(C_start[(i + ctid_y * Y_NUM_PER_THREAD) * N + ctid_x * X_NUM_PER_THREAD + 0]) = FETCH_FLOAT4(sum[i][0]);
        FETCH_FLOAT4(C_start[(i + ctid_y * Y_NUM_PER_THREAD) * N + ctid_x * X_NUM_PER_THREAD + 4]) = FETCH_FLOAT4(sum[i][4]);
    }
}

template __global__ void sgemm7<128, 128, 8, 8, 8>(float *A, float *B, float *C, const int M, const int N, const int K);

__device__ __forceinline__ void reIndex(int *new_index, int x_dim)
{
    new_index[0] = threadIdx.x % x_dim;
    new_index[1] = threadIdx.x / x_dim;
}