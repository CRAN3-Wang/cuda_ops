#include <cstdio>
#include <cuda.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define THREAD_PER_BLOCK 256
#define WARP_SIZE (32)

__global__ void reduce0(float *d_input, float *d_output)
{
    unsigned int tid = threadIdx.x;
    unsigned int global_tid = blockIdx.x * blockDim.x + tid;

    for (unsigned int i = 1; i < blockDim.x; i *= 2)
    {
        if (tid % (i * 2) == 0)
        {
            d_input[global_tid] += d_input[global_tid + i];
            __syncthreads();
        }
    }

    if (tid == 0)
    {
        d_output[blockIdx.x] = d_input[global_tid];
    }
}

__global__ void reduce1(float *d_input, float *d_output)
{
    __shared__ float shared_mem[THREAD_PER_BLOCK];
    unsigned int tid = threadIdx.x;
    unsigned int global_tid = blockIdx.x * blockDim.x + tid;

    shared_mem[tid] = d_input[global_tid];
    __syncthreads();

    for (unsigned int i = 1; i < blockDim.x; i *= 2)
    {
        if (tid % (i * 2) == 0)
        {
            shared_mem[tid] += shared_mem[tid + i];
            __syncthreads();
        }
    }

    if (tid == 0)
    {
        d_output[blockIdx.x] = shared_mem[tid];
    }
}

__global__ void reduce2(float *d_input, float *d_output)
{
    __shared__ float shared_mem[THREAD_PER_BLOCK];
    unsigned int tid = threadIdx.x;
    unsigned int global_tid = blockIdx.x * blockDim.x + tid;

    shared_mem[tid] = d_input[global_tid];
    __syncthreads();

    for (unsigned int i = 1; i < blockDim.x; i *= 2)
    {
        if (tid < blockDim.x / (i * 2))
        {
            shared_mem[2 * i * tid] += shared_mem[2 * i * tid + i];
            __syncthreads();
        }
    }

    if (tid == 0)
    {
        d_output[blockIdx.x] = shared_mem[tid];
    }
}

__global__ void reduce3(float *d_input, float *d_output)
{
    __shared__ float shared_mem[THREAD_PER_BLOCK];
    unsigned int tid = threadIdx.x;
    unsigned int global_tid = blockIdx.x * blockDim.x + tid;

    shared_mem[tid] = d_input[global_tid];
    __syncthreads();

    for (unsigned int i = 1; i < blockDim.x; i *= 2)
    {
        if (tid < blockDim.x / (i * 2))
        {
            shared_mem[tid] += shared_mem[tid + blockDim.x / (i * 2)];
            __syncthreads();
        }
    }

    if (tid == 0)
    {
        d_output[blockIdx.x] = shared_mem[tid];
    }
}

__device__ void warpReduce(volatile float *cache, unsigned int tid)
{
    cache[tid] += cache[tid + 32];
    cache[tid] += cache[tid + 16];
    cache[tid] += cache[tid + 8];
    cache[tid] += cache[tid + 4];
    cache[tid] += cache[tid + 2];
    cache[tid] += cache[tid + 1];
}

__global__ void reduce5(float *d_input, float *d_output)
{
    __shared__ float shared_mem[THREAD_PER_BLOCK];
    unsigned int tid = threadIdx.x;
    unsigned int global_tid = 2 * blockIdx.x * blockDim.x + tid;

    shared_mem[tid] = d_input[global_tid] + d_input[global_tid + blockDim.x];
    __syncthreads();

    // Division is not effective, using bit-ops. And we unfold the last loop to reduce the sync time.
    for (unsigned int i = blockDim.x / 2; i > 32; i >>= 1)
    {
        if (tid < i)
        {
            shared_mem[tid] += shared_mem[tid + i];
            __syncthreads();
        }
    }

    if (tid < 32)
        warpReduce(shared_mem, tid);
    if (tid == 0)
        d_output[blockIdx.x] = shared_mem[tid];
}

__global__ void reduce6(float *d_input, float *d_output)
{
    __shared__ float shared_mem[THREAD_PER_BLOCK];
    unsigned int tid = threadIdx.x;
    unsigned int global_tid = 2 * blockIdx.x * blockDim.x + tid;

    shared_mem[tid] = d_input[global_tid] + d_input[global_tid + blockDim.x];
    __syncthreads();

    if(THREAD_PER_BLOCK >= 512){
        if(tid < 256) shared_mem[tid] += shared_mem[tid + 256];
        __syncthreads();
    }

    if(THREAD_PER_BLOCK >= 256){
        if(tid < 128) shared_mem[tid] += shared_mem[tid + 128];
        __syncthreads();
    }

    if(THREAD_PER_BLOCK >= 128){
        if(tid < 64) shared_mem[tid] += shared_mem[tid + 64];
        __syncthreads();
    }

    if (tid < 32) warpReduce(shared_mem, tid);

    if (tid == 0) d_output[blockIdx.x] = shared_mem[tid];
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

    int block_num = N / THREAD_PER_BLOCK;
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
        for (int j = 0; j < THREAD_PER_BLOCK; ++j)
        {
            res[i] += input[i * THREAD_PER_BLOCK + j];
        }
    }

    cudaMemcpy(d_input, input, N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 Grid(block_num, 1);
    dim3 Block(THREAD_PER_BLOCK, 1);
    reduce0<<<Grid, Block>>>(d_input, d_output);
    reduce1<<<Grid, Block>>>(d_input, d_output);
    reduce2<<<Grid, Block>>>(d_input, d_output);
    reduce3<<<Grid, Block>>>(d_input, d_output);
    reduce5<<<Grid, Block>>>(d_input, d_output);
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