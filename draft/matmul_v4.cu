#include<iostream>
#include<vector>
#include"cuda_runtime.h"
#include<cublas_v2.h>

#define FETCH_FLOAT4(ptr) (reinterpret_cast<float4*>(&(ptr))[0])
#define OFFSET(row, col, N) ((row) * (N) + (col))
template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void sgemm_v4(float* A, float* B, float* C, int M, int N, int K, float alpha, float beta){
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    const int bx = blockIdx.x;
    const int by = blockIdx.y;

    
}