#include <cstdio>
#include <cuda.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define THREAD_PER_BLOCK 256

__device__ void warpReduce(volatile float *cache, unsigned int tid)
{
    if (THREAD_PER_BLOCK > 64)
        cache[tid] += cache[tid + 32];
    if (THREAD_PER_BLOCK > 32)
        cache[tid] += cache[tid + 16];
    if (THREAD_PER_BLOCK > 16)
        cache[tid] += cache[tid + 8];
    if (THREAD_PER_BLOCK > 8)
        cache[tid] += cache[tid + 4];
    if (THREAD_PER_BLOCK > 4)
        cache[tid] += cache[tid + 2];
    if (THREAD_PER_BLOCK > 2)
        cache[tid] += cache[tid + 1];
}

__global__ void reduce6(float *d_input, float *d_output)
{
    __shared__ float shared_mem[THREAD_PER_BLOCK];
    unsigned int tid = threadIdx.x;
    unsigned int global_tid = 2 * blockIdx.x * blockDim.x + tid;

    shared_mem[tid] = d_input[global_tid] + d_input[global_tid + blockDim.x];
    __syncthreads();

    // Unroll all the iterations by #pragma unroll or manually unroll
    // for (unsigned int i = blockDim.x / 2; i > 32; i >>= 1)
    // {
    //     if (tid < i)
    //     {
    //         shared_mem[tid] += shared_mem[tid + i];
    //         __syncthreads();
    //     }
    // }

    if (THREAD_PER_BLOCK >= 512)
    {
        if (tid < 256)
            shared_mem[tid] += shared_mem[tid + 256];
        __syncthreads();
    }

    if (THREAD_PER_BLOCK >= 256)
    {
        if (tid < 128)
            shared_mem[tid] += shared_mem[tid + 128];
        __syncthreads();
    }

    if (THREAD_PER_BLOCK >= 128)
    {
        if (tid < 64)
            shared_mem[tid] += shared_mem[tid + 64];
        __syncthreads();
    }

    if (tid < 32)
        warpReduce(shared_mem, tid);

    if (tid == 0)
        d_output[blockIdx.x] = shared_mem[tid];
}

bool check(float *output, float *res, int n)
{
    for (int i = 0; i < n; ++i)
    {
        if (abs(output[i] - res[i]) > 0.0005)
            return false;
    }
    return true;
}

int main()
{
    const int N = 32 * 1024 * 1024;
    float *input = (float *)malloc(N * sizeof(float));
    float *d_input;
    // Use cudaMalloc to alloc memory on gpu. Using double ptr for DIRECTLY modify the original d_input(make this address to point to the alloc'd gpu mem) more or less like using inference.
    cudaMalloc((void **)&d_input, N * sizeof(N));

    // We will add one epoch when loading, conseqeuntly the block num reduced by half.
    int block_num = N / THREAD_PER_BLOCK / 2;
    float *output = (float *)malloc(block_num * sizeof(float));
    float *d_output;
    cudaMalloc((void **)&d_output, block_num * sizeof(float));

    float *res = (float *)malloc(block_num * sizeof(float));

    for (int i = 0; i < N; ++i)
    {
        input[i] = drand48();
    }

    // Using cpu compute naive reduce result
    for (int i = 0; i < block_num; ++i)
    {
        for (int j = 0; j < 2 * THREAD_PER_BLOCK; ++j)
        {
            res[i] += input[i * 2 * THREAD_PER_BLOCK + j];
        }
    }

    cudaMemcpy(d_input, input, N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 Grid(block_num, 1);
    dim3 Block(THREAD_PER_BLOCK, 1);
    reduce6<<<Grid, Block>>>(d_input, d_output);

    cudaMemcpy(output, d_output, block_num * sizeof(float), cudaMemcpyDeviceToHost);

    if (check(output, res, block_num))
        printf("Correct!\n");
    else
    {
        printf("Incorrect!\n");
        for (int i = 0; i < block_num; ++i)
        {
            if (output[i] != res[i])
            {
                printf("Incorrect element: %lf (exp: %lf) in index %i\n", output[i], res[i], i);
            }
        }
    }

    cudaFree(d_input);
    cudaFree(d_output);
}