#include "sgemv_utils.hpp"
#include <cstdio>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <math.h>

#define eps 0.005

void randomMatrix(const int m, const int n, float *A)
{
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            A[i * n + j] = 2.0 * (float)drand48() - 1.0;
        }
    }
}

void randomVector(const int n, float *x)
{
    for (int i = 0; i < n; i++)
    {
        x[i] = 2.0 * (float)drand48() - 1.0;
    }
}

void cublas_sgemv(float *d_A, float *d_x, float *d_y, const int m, const int n)
{
    cublasHandle_t blas_handle;
    cublasCreate(&blas_handle);
    float alpha = 1.0f;
    float beta = 0.0f;

    cublasSgemv(blas_handle, CUBLAS_OP_T, n, m, &alpha, d_A, n, d_x, 1, &beta, d_y, 1);

    cublasDestroy(blas_handle);
}

void check(float *h_mysgemv_y, float *h_cublas_y, const int m)
{
    for (int i = 0; i < m; i++)
    {
        float diff = abs(h_mysgemv_y[i] - h_cublas_y[i]);
        if (diff > eps)
        {
            printf("Incorrect val: %lf(exp: %lf) at %d\n", h_mysgemv_y[i], h_cublas_y[i], i);
            return;
        }
    }
    printf("Correct!\n");
}