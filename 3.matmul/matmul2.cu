#include<iostream>
#include"cuda_runtime.h"
#include<vector>
#include <cublas_v2.h>



template <const int BLOCKSIZE>

__global__ void mysgemm_v2(int M, int N, int K, float alpha, float* A, float* B, float beta, float* C){
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // 为每个线程块分配一个矩阵中的小分块进行处理
    const int BM = BLOCKSIZE;
    const int BN = BLOCKSIZE;
    const int BK = BLOCKSIZE;

    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];
    
    int tx = threadIdx.x % BN;
    int ty = threadIdx.x / BN;

    A = &A[by * BM * BK];
    B = &B[bx * BN];
    C = &C[by * BM * N + bx * BN];

    float temp = 0.f;
    for (int k = 0; k < K; k += BK){
        As[ty * BK + tx] = A[ty * K + tx];
        Bs[ty * BN + tx] = B[ty * N + tx];

        __syncthreads();

        A += BK;
        B += BK * N;

        for (int i = 0; i < BK; i++){
            temp += As[ty * BK + i] * Bs[i * BN + tx];
        }

        __syncthreads();

        C[ty * N + tx] = alpha * temp + beta * C[ty * N + tx];
    }
}

int main(){
    std::vector<int> sizes = {128, 256, 512, 1024, 2048, 4096, 8192};
    for (int N:sizes){
        size_t elemCount = N * N;
        std::cout<<"------------------------Testing size: "<< N <<"------------------------"<< std::endl;

        size_t numBytes = elemCount * sizeof(float);
        float* A = (float*)malloc(numBytes);
        float* B = (float*)malloc(numBytes);
        float* C_cublas = (float*)malloc(numBytes);
        float* C_v1 = (float*)malloc(numBytes);

        float* d_A,* d_B, * d_C_v1;
        cudaMalloc(&d_A, numBytes);
        cudaMalloc(&d_B, numBytes);
        cudaMalloc(&d_C_v1, numBytes);

        try{
            for (int i = 0; i < elemCount; i++){
                A[i] = 1.0f;
                B[i] = 2.0f;
            }

            cudaMemcpy(d_A, A, numBytes, cudaMemcpyHostToDevice);
            cudaMemcpy(d_B, B, numBytes, cudaMemcpyHostToDevice);
            
            cublasHandle_t handle;  //定义句柄, 用于记录状态
            cublasCreate(&handle);  // 创建cuBLAS上下文

            float alpha = 1.f;
            float beta = 0.f;
            

            /*------------------------cublas计算/*------------------------*/
            cudaEvent_t start, end;
            cudaEventCreate(&start);
            cudaEventCreate(&end);

            int warmup_time = 10;
            for (int i = 0; i < warmup_time; i++){
                cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_B, N, d_A, N, &beta, d_C_v1, N);
            }

            cudaDeviceSynchronize();

            int repeat_time = 5;
            cudaEventRecord(start);
            for (int i = 0; i < repeat_time; i++){
                cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_B, N, d_A, N, &beta, d_C_v1, N);
            }
            cudaEventRecord(end);
            cudaEventSynchronize(end);

            float cublas_time = 0;
            cudaEventElapsedTime(&cublas_time, start, end);
            
            cudaMemcpy(C_cublas, d_C_v1, numBytes, cudaMemcpyDeviceToHost);
            std::cout<<"cublas time:"<<cublas_time<<"ms"<<std::endl;

            /*------------------------v1计算------------------------*/
            dim3 threads(1024); // 1024拆成了32x32
            dim3 blocks((N + 32 - 1) / 32, (N + 32 - 1) / 32);
            
            for (int i = 0; i < warmup_time; i++){
                mysgemm_v2<32><<<blocks, threads>>>(N, N, N, alpha, d_A, d_B, beta, d_C_v1);
            }
            cudaDeviceSynchronize();

            cudaEventRecord(start);
            for (int i = 0; i < repeat_time; i++){
                mysgemm_v2<32><<<blocks, threads>>>(N, N, N, alpha, d_A, d_B, beta, d_C_v1);
            }
            cudaEventRecord(end);
            cudaEventSynchronize(end);

            float v1_time = 0.f;
            cudaEventElapsedTime(&v1_time, start, end);

            cudaMemcpy(C_v1, d_C_v1, numBytes, cudaMemcpyDeviceToHost);
            std::cout<<"v2 time:"<<v1_time<<"ms"<<std::endl;

            
            // 结果比较
            bool isMatch = true;
            for (int i = 0; i < elemCount; i++){
                if (fabsf(C_cublas[i] - C_v1[i]) > 1e-3){
                    isMatch = false;
                    break;
                }
            }
            if (isMatch) std::cout<<"Results Match!"<<std::endl;
            else std::cout<<"Results not Match!"<<std::endl;

            
        }
        catch(...){
            std::cerr << "Out of memory or error during testing size: " << N << std::endl;
        }
    }
    
}