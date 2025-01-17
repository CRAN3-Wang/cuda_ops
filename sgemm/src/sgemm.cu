#include "../include/utils.hpp"
#include "../include/sgemm_v0_global_mem.cuh"
#include <cstdio>
#include <cuda.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define THREAD_PER_BLOCK (16)

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
    cudaMalloc((void**)&d_A, memsize_A);
    cudaMalloc((void**)&d_B, memsize_B);
    cudaMalloc((void**)&d_C, memsize_C);

    cudaMemcpy(d_A, h_A, memsize_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, memsize_B, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(THREAD_PER_BLOCK, THREAD_PER_BLOCK);
    dim3 blocksPerGrid((n + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK, (m + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK);

    sgemm0<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, m, n, k);

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