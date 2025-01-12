#include <cstdio>
#include <cuda.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define THREAD_PER_BLOCK (64)

void __global__ add1(float *A, float *B, float *C)
{
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    C[global_tid] = A[global_tid] + B[global_tid];
}

void __global__ add2(float *A, float *B, float *C)
{
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x + 1; //+1 offset, un-aligned memory layout
    C[global_tid] = A[global_tid] + B[global_tid];
}

void __global__ add3(float *A, float *B, float *C)
{
    int permuted_tid = threadIdx.x ^ 0x1;
    int global_tid = blockIdx.x * blockDim.x + permuted_tid;
    C[global_tid] = A[global_tid] + B[global_tid];
}

void __global__ add4(float *A, float *B, float *C)
{
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warpId = global_tid >> 5;
    C[warpId] = A[warpId] + B[warpId];
}

void __global__ add5(float *A, float *B, float *C)
{
    int global_tid = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    C[global_tid] = A[global_tid] + B[global_tid];
}

int main()
{
    const int N = 32 * 1024 * 1024;
    float *input_A = (float *)malloc(N * sizeof(float));
    float *d_input_A;
    cudaMalloc((void **)&d_input_A, N * sizeof(float));

    float *input_B = (float *)malloc(N * sizeof(float));
    float *d_input_B;
    cudaMalloc((void **)&d_input_B, N * sizeof(float));

    float *output = (float *)malloc(N * sizeof(float));
    float *d_output;
    cudaMalloc((void **)&d_output, N * sizeof(float));

    dim3 Grid(N / 256);
    dim3 Block(THREAD_PER_BLOCK, 1);

    add1<<<Grid, Block>>>(d_input_A, d_input_B, d_output);
    add2<<<Grid, Block>>>(d_input_A, d_input_B, d_output);
    add3<<<Grid, Block>>>(d_input_A, d_input_B, d_output);
    add4<<<Grid, Block>>>(d_input_A, d_input_B, d_output);
    add5<<<Grid, Block>>>(d_input_A, d_input_B, d_output);
}