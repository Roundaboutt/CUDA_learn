#include"cuda_runtime.h"
#include<iostream>
#include<chrono>
#include<vector>
#include<cublas_v2.h>

#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*> (&(pointer))[0])
#define OFFSET(row, col, ld) ((row) * (ld) + (col))

template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void sgemm_v4(int M, int N, int K, float* A, float* B, float* C, float alpha, float beta){
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // 行、列分别需要多少线程
    const int block_row_thread = BN / TN;
    const int block_col_thread = BM / TM;
    const int thread_num = block_row_thread * block_col_thread;

    // 每个 tile 要用线程搬运几次(一次搬运一个 float4 )
    const int ldg_a_num = BK * BM / thread_num / 4;     //4 是向量宽度
    const int ldg_b_num = BN * BK / thread_num / 4;

    // 线程在 tile 内的坐标
    int tx = (threadIdx.x % block_row_thread) * TN;
    int ty = (threadIdx.x / block_row_thread) * TM;

    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    int a_tile_row = threadIdx.x / (BK / 4);        // 当前线程负责从 A tile 的第几行开始搬 
    int a_tile_col = threadIdx.x % (BK / 4) * 4;
    int a_tile_stride = BM / ldg_a_num;             // 每个线程每轮搬运的行跨度(一共有 BM 行, 一次搬运 ldg_a_num 行)

    int b_tile_row = threadIdx.x / (BN / 4);
    int b_tile_col = threadIdx.x % (BN / 4) * 4;
    int b_tile_stride = BK / ldg_b_num;

    float accum[TM][TN] = {0.};

    // 临时load缓冲区 每个线程要搬ldg_a_num次,每次搬4个float(float4), 所以大小是4 * ldg_a_num
    // 因为要把 A 的行搬到 As 的列中, 所以需要一个寄存器来暂存 A 的行
    float ldg_a_reg[4 * ldg_a_num] = {0.};

    float a_frag[TM]; // 把 A 的一列放入寄存器缓存
    float b_frag[TN];

    A = &A[by * BM * K];
    B = &B[bx * BN];
    C = &C[by * BM * N + bx * BN];

#pragma unroll
    for (int k = 0; k < K; k += BK){
#pragma unroll
        // 把 A 中的一小块搬运到As
        for (int i = 0; i < BM; i += a_tile_stride){
            int ldg_index = i / a_tile_stride * 4;      // 第 i 轮搬运时起始的索引
            FETCH_FLOAT4(ldg_a_reg[ldg_index]) = FETCH_FLOAT4(A[OFFSET(a_tile_row + i, a_tile_col, K)]);

            // 把A转置放入As  因为最终是 A 的列与 B 的行做点积, 此时 As 的行就是 A 的列, 按行读取会效率更高
            As[OFFSET(a_tile_col, i + a_tile_row, BM)] = ldg_a_reg[ldg_index];
            As[OFFSET(a_tile_col + 1, i + a_tile_row, BM)] = ldg_a_reg[ldg_index + 1];
            As[OFFSET(a_tile_col + 2, i + a_tile_row, BM)] = ldg_a_reg[ldg_index + 2];
            As[OFFSET(a_tile_col + 3, i + a_tile_row, BM)] = ldg_a_reg[ldg_index + 3];
        }

#pragma unroll
        // 把 B 中的一小块搬运到 Bs
        for (int i = 0; i < BK; i += b_tile_stride){
            FETCH_FLOAT4(Bs[OFFSET(b_tile_row + i, b_tile_col, BN)]) = FETCH_FLOAT4(B[OFFSET(b_tile_row + i, b_tile_col, N)]);
        }

        __syncthreads();

        A += BK;
        B += BK * N;

#pragma unroll
        for (int k1 = 0; k1 < BK; k1++){
            for (int i = 0; i < TM; i += 4){
                                                            //   列, 行 因为在上面已经转置过了
                FETCH_FLOAT4(a_frag[i]) = FETCH_FLOAT4(As[OFFSET(k1, ty + i, BM)]);
            }

            for (int i = 0; i < TN; i += 4){
                                                            //  行, 列
                FETCH_FLOAT4(b_frag[i]) = FETCH_FLOAT4(Bs[OFFSET(k1, tx + i, BN)]);
            }

#pragma unroll
            for (int m = 0; m < TM; m++){
#pragma unroll                
                for (int n = 0; n < TN; n++){
                    accum[m][n] += a_frag[m] * b_frag[n];
                }
            }
        }
        __syncthreads();
    }
    
#pragma unroll
    for (int m = 0; m < TM; m++){
#pragma unroll
        for (int n = 0; n < TN; n += 4){
            float4 ctemp = FETCH_FLOAT4(C[OFFSET(ty + m, tx + n, N)]);
            ctemp.x = alpha * accum[m][n] + beta * ctemp.x;
            ctemp.y = alpha * accum[m][n + 1] + beta * ctemp.y;
            ctemp.z = alpha * accum[m][n + 2] + beta * ctemp.z;
            ctemp.w = alpha * accum[m][n + 3] + beta * ctemp.w;
            FETCH_FLOAT4(C[OFFSET(ty + m, tx + n, N)]) = ctemp;
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

            int repeat_time = 10;
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

            /*------------------------v4计算------------------------*/
            dim3 threads(256);
            dim3 blocks((N + 128 - 1) / 128, (N + 128 - 1) / 128);
            
            for (int i = 0; i < warmup_time; i++){
                sgemm_v4<128, 128, 8, 8, 8><<<blocks, threads>>>(N, N, N, d_A, d_B, d_C_sgemm, alpha, beta);
            }
            cudaDeviceSynchronize();

            cudaEventRecord(start);
            for (int i = 0; i < repeat_time; i++){
                sgemm_v4<128, 128, 8, 8, 8><<<blocks, threads>>>(N, N, N, d_A, d_B, d_C_sgemm, alpha, beta);
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