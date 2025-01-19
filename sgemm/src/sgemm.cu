#include "utils.hpp"
#include "sgemm_v0_global_mem.cuh"
#include "sgemm_v1_shared_mem.cuh"
#include "sgemm_v2_increase_workload_of_threads.cuh"
#include "sgemm_v3_float4.cuh"
#include "sgemm_v4_reg.cuh"
#include "sgemm_v5_reg_float4.cuh"
#include <cstdio>
#include <cuda.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define THREAD_PER_BLOCK (16)
#define STRIDE (2)

int main()
{
    int m = 2048;
    int n = 2048;
    int k = 2048;

    size_t memsize_A = m * k * sizeof(float);
    size_t memsize_B = k * n * sizeof(float);
    size_t memsize_C = m * n * sizeof(float);

    float *h_A = (float *)malloc(memsize_A);
    float *h_B = (float *)malloc(memsize_B);
    float *h_C_h = (float *)malloc(memsize_C);
    float *h_C_d = (float *)malloc(memsize_C);

    randomMatrix(m, k, h_A);
    randomMatrix(k, n, h_B);

    cpu_sgemm(h_A, h_B, h_C_h, m, n, k);

    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, memsize_A);
    cudaMalloc((void **)&d_B, memsize_B);
    cudaMalloc((void **)&d_C, memsize_C);

    cudaMemcpy(d_A, h_A, memsize_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, memsize_B, cudaMemcpyHostToDevice);

    // {
    //     dim3 Block(THREAD_PER_BLOCK, THREAD_PER_BLOCK);
    //     dim3 Grid((n + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK, (m + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK);

    //     sgemm0<<<Grid, Block>>>(d_A, d_B, d_C, m, n, k);
    //     sgemm1<THREAD_PER_BLOCK><<<Grid, Block>>>(d_A, d_B, d_C, m, n, k);
    // }

    // {
    //     dim3 Block(THREAD_PER_BLOCK, THREAD_PER_BLOCK);
    //     dim3 Grid((n + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK / STRIDE, (m + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK / STRIDE);
    //     sgemm2<THREAD_PER_BLOCK, STRIDE><<<Grid, Block>>>(d_A, d_B, d_C, m, n, k);
    // }

    // {
    //     dim3 Block(8, 32);
    //     dim3 Grid(m / 32, n / 32);
    //     sgemm3<32, 32, 32, 4><<<Grid, Block>>>(d_A, d_B, d_C, m, n, k);
    // }

    // {
    //     dim3 Block(16 * 16);
    //     dim3 Grid(m / 32, n / 32);
    //     sgemm4<32, 32, 32, 4><<<Grid, Block>>>(d_A, d_B, d_C, m, n, k);
    // }

    {
        constexpr int M_NUM_PER_BLOCK = 64;
        constexpr int N_NUM_PER_BLOCK = 64;
        constexpr int K_NUM_PER_BLOCK = 64;
        constexpr int M_NUM_PER_THREAD = 4;
        constexpr int N_NUM_PER_THREAD = 4;
        constexpr int K_NUM_PER_THREAD = 4;

        dim3 Block(16 * 16);
        dim3 Grid(m / M_NUM_PER_BLOCK, n / N_NUM_PER_BLOCK);
        sgemm5<M_NUM_PER_BLOCK, N_NUM_PER_BLOCK, K_NUM_PER_BLOCK, M_NUM_PER_THREAD, N_NUM_PER_THREAD, K_NUM_PER_THREAD><<<Grid, Block>>>(d_A, d_B, d_C, m, n, k);
    }
    cudaMemcpy(h_C_d, d_C, memsize_C, cudaMemcpyDeviceToHost);

    check(h_C_d, h_C_h, m, n);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C_h);
    free(h_C_d);

    return 0;
}