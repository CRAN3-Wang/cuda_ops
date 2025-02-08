#pragma once

void randomMatrix(const int m, const int n, float *A);
void randomVector(const int n, float *x);
void cublas_sgemv(float *A, float *x, float *y, const int m, const int n);
void check(float *h_mysgemv_y, float *h_cublas_y, const int m);