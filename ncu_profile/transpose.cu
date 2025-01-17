#include <cstdio>
#include <cuda.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define FETCH_FLOAT4(ptr) (reinterpret_cast<float4 *>(&(ptr))[0])
__global__ void cudaTranspose_inner_4x4(float *d_input, float *d_output, const int M, const int N)
{
    float src[4][4];
    float dst[4][4];

    int global_i = blockIdx.y * blockDim.y + threadIdx.y;
    int global_j = blockIdx.x * blockDim.x + threadIdx.x;
    int src_i = (global_i * N + global_j) << 2;
    int dst_i = (global_j * M + global_i) << 2;

    #pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        FETCH_FLOAT4(src[i]) = FETCH_FLOAT4(d_input[src_i + i * N]);
    }

    #pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        FETCH_FLOAT4(dst[i]) = make_float4(src[0][i], src[1][i], src[2][i], src[3][i]);
    }

    #pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        FETCH_FLOAT4(d_output[dst_i + i * M]) = FETCH_FLOAT4(dst[i][0]);
    }
}

__global__ void cudaTranspose_naive(float *d_input, float *d_output, const int M, const int N)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    d_output[j * M + i] = d_input[i * N + j];
}

void cpuTranspose(float *h_input, float *h_output, int M, int N)
{
    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            size_t in_idx = i * N + j;
            size_t out_idx = j * M + i;
            h_output[out_idx] = h_input[in_idx];
        }
    }
}

int main()
{
    int M = 2048;
    int N = 512;
    const size_t size = M * N;

    float *h_input = (float *)malloc(size * sizeof(float));
    float *h_output = (float *)malloc(size * sizeof(float));
    float *res = (float *)malloc(size * sizeof(float));

    for (int i = 0; i < size; ++i)
    {
        h_input[i] = 2.0 * drand48() - 1.0;
    }

    cpuTranspose(h_input, res, M, N);

    float *d_input, *d_output;
    cudaMalloc((void **)&d_input, size * sizeof(float));
    cudaMemcpy(d_input, h_input, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc((void **)&d_output, size * sizeof(float));
    cudaMemset(d_output, 0, size * sizeof(float));

    for (int i = 0; i < 2; ++i)
    {
        dim3 Block(32, 8);
        dim3 Grid((N - 1) / Block.x + 1, (M - 1) / Block.y + 1);

        cudaTranspose_naive<<<Grid, Block>>>(d_input, d_output, M, N);
        cudaDeviceSynchronize();
    }

    for (int i = 0; i < 2; ++i)
    {
        dim3 Block(16, 16);
        dim3 Grid((N - 1) / Block.x + 1, (M - 1) / Block.y + 1);

        cudaTranspose_naive<<<Grid, Block>>>(d_input, d_output, M, N);
        cudaDeviceSynchronize();
    }

    for (int i = 0; i < 2; ++i)
    {
        dim3 Block(8, 32);
        dim3 Grid((N - 1) / Block.x + 1, (M - 1) / Block.y + 1);

        cudaTranspose_naive<<<Grid, Block>>>(d_input, d_output, M, N);
        cudaDeviceSynchronize();
    }

    for (int i = 0; i < 2; ++i)
    {
        dim3 Block(32, 8);
        dim3 Grid(((N - 1) / Block.x + 1) >> 2, ((M - 1) / Block.y + 1) >> 2);

        cudaTranspose_inner_4x4<<<Grid, Block>>>(d_input, d_output, M, N);
        cudaDeviceSynchronize();
    }

    for (int i = 0; i < 2; ++i)
    {
        dim3 Block(16, 16);
        dim3 Grid(((N - 1) / Block.x + 1) >> 2, ((M - 1) / Block.y + 1) >> 2);

        cudaTranspose_inner_4x4<<<Grid, Block>>>(d_input, d_output, M, N);
        cudaDeviceSynchronize();
    }

    cudaMemcpy(h_output, d_output, size * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < M; ++j)
        {
            size_t idx = i * M + j;
            if (res[idx] != h_output[idx])
            {
                printf("Wrong element in row (%d, %d), host: %lf, device: %lf", i, j, res[idx], h_output[idx]);
            }
        }
    }
}