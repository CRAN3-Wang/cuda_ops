#include <cstdio>
#include <cuda.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define THREAD_PER_BLOCK (256)
#define WARP_SIZE (32)

__device__ __forceinline__ float warpReduce(float sum)
{
    // Here we use register of each warp to perform reduction. The register is commutable for threads in one warp
    if (blockDim.x >= 32)
        sum += __shfl_down_sync(0xffffffff, sum, 16);
    if (blockDim.x >= 16)
        sum += __shfl_down_sync(0xffffffff, sum, 8);
    if (blockDim.x >= 8)
        sum += __shfl_down_sync(0xffffffff, sum, 4);
    if (blockDim.x >= 4)
        sum += __shfl_down_sync(0xffffffff, sum, 2);
    if (blockDim.x >= 2)
        sum += __shfl_down_sync(0xffffffff, sum, 1);
    return sum;
}

template <unsigned int NUM_PER_THREAD>
__global__ void reduce7(float *d_input, float *d_output)
{
    float sum = 0;
    unsigned int tid = threadIdx.x;
    unsigned int global_tid = blockIdx.x * (blockDim.x * NUM_PER_THREAD) + tid;

#pragma unroll
    for (int i = 0; i < NUM_PER_THREAD; ++i)
    {
        sum += d_input[global_tid + i * blockDim.x];
    }

    __syncthreads();

    static __shared__ float warpLevelSums[WARP_SIZE]; // Why is 32? due to thread limitation of a block or?
    const int laneId = tid % WARP_SIZE;
    const int warpId = tid / WARP_SIZE;

    sum = warpReduce(sum);

    // Store the first value of each warp register to block shared mem
    if (laneId == 0)
        warpLevelSums[warpId] = sum;
    __syncthreads();

    // We choose several threads to load the result of each warp reduction. N.B. typical pattern will not cause warp divergence.
    sum = (tid < THREAD_PER_BLOCK / WARP_SIZE) ? warpLevelSums[laneId] : 0;
    // Final reduction
    if (warpId == 0)
        sum = warpReduce(sum);

    if (tid == 0)
        d_output[blockIdx.x] = sum;
}

bool check(float *output, float *res, int n)
{
    for (int i = 0; i < n; ++i)
    {
        if (abs(output[i] - res[i]) > 0.005)
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
    cudaMalloc((void **)&d_input, N * sizeof(float));

    // We will add one epoch when loading, consequently the block num reduced by half.
    constexpr unsigned int block_num = 1024;
    constexpr unsigned int NUM_PER_BLOCK = N / block_num;
    constexpr unsigned int NUM_PER_THREAD = NUM_PER_BLOCK / THREAD_PER_BLOCK;
    float *output = (float *)malloc(block_num * sizeof(float));
    float *d_output;
    cudaMalloc((void **)&d_output, block_num * sizeof(float));

    float *res = (float *)malloc(block_num * sizeof(float));

    for (int i = 0; i < N; ++i)
    {
        input[i] = 2.0 * (float)drand48() - 1.0;
    }

    // Using cpu compute naive reduce result
    for (int i = 0; i < block_num; i++)
    {
        float cur = 0;
        for (int j = 0; j < NUM_PER_BLOCK; j++)
        {
            if (i * NUM_PER_BLOCK + j < N)
            {
                cur += input[i * NUM_PER_BLOCK + j];
            }
        }
        res[i] = cur;
    }

    cudaMemcpy(d_input, input, N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 Grid(block_num, 1);
    dim3 Block(THREAD_PER_BLOCK, 1);
    reduce7<NUM_PER_THREAD><<<Grid, Block>>>(d_input, d_output);

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

    free(input);
    free(output);
    free(res);
}