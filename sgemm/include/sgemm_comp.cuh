#pragma once

template <
    const int BLOCK_SIZE_M,         // height of block of C that each thread block calculate
    const int BLOCK_SIZE_K,         // width of block of A that each thread block load into shared memory
    const int BLOCK_SIZE_N,         // width of block of C that each thread block calculate
    const int THREAD_SIZE_Y,        // height of block of C that each thread calculate
    const int THREAD_SIZE_X,        // width of block of C that each thread calculate
    const bool ENABLE_DOUBLE_BUFFER // whether enable double buffering or not
    >
__global__ void Sgemm(
    float *__restrict__ A,
    float *__restrict__ B,
    float *__restrict__ C,
    const int M,
    const int N,
    const int K);

template <const int BLOCK_SIZE_M,  // height of block of C that each thread block calculate
          const int BLOCK_SIZE_N,  // width of block of A that each thread block load into shared memory
          const int BLOCK_SIZE_K,  // width of block of C that each thread block calculate
          const int THREAD_SIZE_Y, // height of block of C that each thread calculate
          const int THREAD_SIZE_X, // width of block of C that each thread calculate
          const bool ENABLE_DOUBLE_BUFFER>
__global__ void cuda_sgemm(float *A_ptr, float *B_ptr, float *C_ptr, const int M, const int N, const int K);