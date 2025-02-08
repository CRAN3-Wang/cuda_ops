#include "sgemv_utils.hpp"
#include "sgemv_v0_32.cuh"
#include "sgemv_v1_float4.cuh"
#include "sgemv_v2_16.cuh"
#include <cstdio>
#include <cuda.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

int main()
{
    int m = 256;
    int n = 128;

    size_t memsize_A = m * n * sizeof(float);
    size_t memsize_x = n * sizeof(float);
    size_t memsize_y = m * sizeof(float);

    float *h_A = (float *)malloc(memsize_A);
    float *h_x = (float *)malloc(memsize_x);
    float *h_mysgemv_y = (float *)malloc(memsize_y);
    float *h_cublas_y = (float *)malloc(memsize_y);

    randomMatrix(m, n, h_A);
    randomVector(n, h_x);

    float *d_A;
    float *d_x;
    float *d_mysgemv_y;
    float *d_cublas_y;

    cudaMalloc((void **)&d_A, memsize_A);
    cudaMalloc((void **)&d_x, memsize_x);
    cudaMalloc((void **)&d_mysgemv_y, memsize_y);
    cudaMalloc((void **)&d_cublas_y, memsize_y);

    cudaError_t err;

    err = cudaMemcpy(d_A, h_A, memsize_A, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        printf("Failed to copy data from host to device for d_A: %s\n", cudaGetErrorString(err));
        return -1;
    }
    err = cudaMemcpy(d_x, h_x, memsize_x, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        printf("Failed to copy data from host to device for d_A: %s\n", cudaGetErrorString(err));
        return -1;
    }

    dim3 Grid(m / 4);
    dim3 Block(32, 4);

    // sgemv0<<<Grid, Block>>>(d_A, d_x, d_mysgemv_y, m, n);
    sgemv1<<<Grid, Block>>>(d_A, d_x, d_mysgemv_y, m, n);
    // sgemv2<2><<<Grid, Block>>>(d_A, d_x, d_mysgemv_y, m, n);
    cudaMemcpy(h_mysgemv_y, d_mysgemv_y, memsize_y, cudaMemcpyDeviceToHost);
    
    cublas_sgemv(d_A, d_x, d_cublas_y, m, n);
    cudaMemcpy(h_cublas_y, d_cublas_y, memsize_y, cudaMemcpyDeviceToHost);

    check(h_mysgemv_y, h_cublas_y, m);

    cudaFree(d_A);
    cudaFree(d_x);
    cudaFree(d_mysgemv_y);
    cudaFree(d_cublas_y);
    free(h_A);
    free(h_x);
    free(h_mysgemv_y);
    free(h_cublas_y);

    return 0;
}