#include<iostream>
#include<chrono>
#include"cuda_runtime.h"
#include<cublas_v2.h>
#include<vector>

template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void sgemm_v3(int M, int N, int K, float* A, float* B, float* C, float alpha, float beta){
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // 一个block内的线程数
    // 每个线程负责处理一个 TM * TN 大小的数据tile
    int block_row_thread = BN / TN;
    int block_col_thread = BM / TM;
    int thread_num = block_row_thread * block_col_thread;

    // 当前线程在块内计算的起始地址
    int ty = (threadIdx.x / block_row_thread) * TM;
    int tx = (threadIdx.x % block_row_thread) * TN;

    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    A = &A[by * K * BM];
    B = &B[bx * BN];
    C = &C[BM * N * by + bx * BN];

    
    int a_tile_row = threadIdx.x / BK;      // 当前线程要搬到As的第几行
    int a_tile_col = threadIdx.x % BK;
    int a_tile_stride = thread_num / BK;    // 一个线程在for循环里负责不止一行, 要隔stride行再搬

    int b_tile_row = threadIdx.x / BN;      // 当前线程要搬到Bs的第几行
    int b_tile_col = threadIdx.x % BN;
    int b_tile_stride = thread_num / BN;

    float temp[TM][TN] = {0.};

#pragma unroll
    for (int k = 0; k < K; k += BK){

#pragma unroll
        for (int i = 0; i < BM; i += a_tile_stride){
            As[(a_tile_row + i) * BK + a_tile_col] = A[(a_tile_row + i) * K + a_tile_col];
        }

#pragma unroll
        for (int i = 0; i < BK; i += b_tile_stride){
            Bs[(b_tile_row + i) * BN + b_tile_col] = B[(b_tile_row + i) * N + b_tile_col];
        }        

        __syncthreads();
        A += BK;
        B += BK * N;

#pragma unroll
        for (int k1 = 0; k1 < BK; k1++){
#pragma unroll
            for (int i = 0; i < TM ; i++){
                for (int j = 0; j < TN; j++){
                    temp[i][j] += As[(ty + i) * BK + k1] * Bs[tx + j + BN * k1];
                }
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < TM; i++){
        for (int j = 0; j < TN; j++){
            C[(ty + i) * N + tx + j] = alpha * temp[i][j] + beta * C[(ty + i) * N + tx + j];
        }
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
        float* C_sgemm = (float*)malloc(numBytes);

        float* d_A,* d_B, * d_C_sgemm;
        cudaMalloc(&d_A, numBytes);
        cudaMalloc(&d_B, numBytes);
        cudaMalloc(&d_C_sgemm, numBytes);

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
            

            /*------------------------cublas计算------------------------*/
            cudaEvent_t start, end;
            cudaEventCreate(&start);
            cudaEventCreate(&end);

            int warmup_time = 5;
            for (int i = 0; i < warmup_time; i++){
                cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_B, N, d_A, N, &beta, d_C_sgemm, N);
            }

            cudaDeviceSynchronize();

            int repeat_time = 5;
            cudaEventRecord(start);
            for (int i = 0; i < repeat_time; i++){
                cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_B, N, d_A, N, &beta, d_C_sgemm, N);
            }
            cudaEventRecord(end);
            cudaEventSynchronize(end);

            float cublas_time = 0;
            cudaEventElapsedTime(&cublas_time, start, end);
            
            cudaMemcpy(C_cublas, d_C_sgemm, numBytes, cudaMemcpyDeviceToHost);
            std::cout<<"cublas time:"<< cublas_time <<"ms"<<std::endl;

            /*------------------------v3计算------------------------*/
            dim3 threads(256);
            dim3 blocks((N + 128 - 1) / 128, (N + 128 - 1) / 128);
            
            for (int i = 0; i < warmup_time; i++){
                sgemm_v3<128, 128, 8, 8, 8><<<blocks, threads>>>(N, N, N, d_A, d_B, d_C_sgemm, alpha, beta);
            }
            cudaDeviceSynchronize();

            cudaEventRecord(start);
            for (int i = 0; i < repeat_time; i++){
                sgemm_v3<128, 128, 8, 8, 8><<<blocks, threads>>>(N, N, N, d_A, d_B, d_C_sgemm, alpha, beta);
            }
            cudaEventRecord(end);
            cudaEventSynchronize(end);

            float v3_time = 0.f;
            cudaEventElapsedTime(&v3_time, start, end);

            cudaMemcpy(C_sgemm, d_C_sgemm, numBytes, cudaMemcpyDeviceToHost);
            std::cout<<"v3 time:"<< v3_time <<"ms"<<std::endl;

            std::cout<< "sgemm' GFLOPS is "<< cublas_time / v3_time * 100 << "% of cublas" << std::endl; 
            
            // 结果比较
            bool isMatch = true;
            for (int i = 0; i < elemCount; i++){
                if (fabsf(C_cublas[i] - C_sgemm[i]) > 1e-5){
                    isMatch = false;
                    break;
                }
            }
            if (isMatch) std::cout<<"Results Match!"<<std::endl;
            else std::cout<<"Results not Match!"<<std::endl;

            
        }
        catch(...){
            std::cerr << "Error testing size: " << N << std::endl;
        }
    }
    
}