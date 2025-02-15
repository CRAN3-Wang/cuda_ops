#include "sgemm_utils.hpp"
#include <cstdio>
#include <stdlib.h>
#include <cblas.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define eps 0.005

void randomMatrix(const int m, const int n, float *a)
{
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            a[i * n + j] = 2.0 * (float)drand48() - 1.0;
        }
    }
}

void cpu_sgemm(float *A, float *B, float *C, const int M, const int N, const int K)
{
    const float alpha = 1.0f;
    const float beta = 0.0f;

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, N, K, alpha, A, K, B, N, beta, C, N);
}

void cublas_sgemm(float *d_A, float *d_B, float *d_C, const int M, const int N, const int K) {
    cublasHandle_t blas_handle;
    cublasCreate(&blas_handle);
    float alpha = 1.0f;
    float beta = 0.0f;

    cublasSgemm(blas_handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha, d_A, K, d_B, N, &beta, d_C, N);

    cublasDestroy(blas_handle);
}

void check(float *device, float *host, const int m, const int n)
{
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            float diff = abs(device[i * n + j] - host[i * n + j]);
            if (diff > eps)
            {
                printf("Incorrect val: %lf(exp: %lf) at %d, %d\n", device[i * n + j], host[i * n + j], i, j);
                return;
            }
        }
    }
    printf("Correct!\n");
}