#pragma once

void randomMatrix(const int m, const int n, float *a);
void cpu_sgemm(float *A, float *B, float *C, const int M, const int N, const int K);
void check(float *device, float *host, const int m, const int n);