#include<iostream>
#include<chrono>
#include"cuda_runtime.h"
#include<cublas_v2.h>
#include<vector>


#define OFFSET(row, col ,ld) ((row) * (ld) + (col))
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__  void __launch_bounds__(256) mysgemm_v5(int M, int N, int K, float alpha, float* A, float* B, float beta, float* C){
    const int bx = blockIdx.x;
    const int by = blockIdx.y;

    const int block_row_thread = BN / TN;
    const int block_col_thread = BM / TM;
    const int num_threads = block_row_thread * block_col_thread;

    const int ldg_a_num = BK * BM / num_threads / 4;
    const int ldg_b_num = BN * BK / num_threads / 4;

    const int tx = (threadIdx.x % block_row_thread) * TN;
    const int ty = (threadIdx.x / block_row_thread) * TM;

    __shared__ float As[2][BK * BM];
    __shared__ float Bs[2][BK * BN];

    const int a_tile_row = threadIdx.x / (BK / 4);
    const int a_tile_col = threadIdx.x % (BK / 4) * 4;
    const int a_tile_stride = BM / ldg_a_num;

    const int b_tile_row = threadIdx.x / (BN / 4);
    const int b_tile_col = threadIdx.x % (BN / 4) * 4;
    const int b_tile_stride = BK / ldg_b_num;

    float accum[TM][TN] = {0.f};

    A = &A[by * BM * K];
    B = &B[bx * BN];
    C = &C[by * BM * N + bx * BN];

    float a_frag[2][TM];
    float b_frag[2][TN];

    float ldg_a_reg[4 * ldg_a_num] = {0.f};
    float ldg_b_reg[4 * ldg_b_num] = {0.f};

    // 先把 A 的第一个小块(列)转置搬到 As[0]    k = 0
#pragma unroll    
    for (int i = 0; i < BM; i += a_tile_stride){
        int ldg_index = i / a_tile_stride * 4;
        FETCH_FLOAT4(ldg_a_reg[ldg_index]) = FETCH_FLOAT4(A[OFFSET(a_tile_row + i, a_tile_col, K)]);
        As[0][OFFSET(a_tile_col, a_tile_row + i, BM)] = ldg_a_reg[ldg_index];
        As[0][OFFSET(a_tile_col + 1, a_tile_row + i, BM)] = ldg_a_reg[ldg_index + 1];
        As[0][OFFSET(a_tile_col + 2, a_tile_row + i, BM)] = ldg_a_reg[ldg_index + 2];
        As[0][OFFSET(a_tile_col + 3, a_tile_row + i, BM)] = ldg_a_reg[ldg_index + 3];
    }
    
    // 先把 B 的第一个小块(行)搬到Bs[0]    k = 0
#pragma unroll    
    for (int i = 0; i < BK; i += b_tile_stride){
        FETCH_FLOAT4(Bs[0][OFFSET(b_tile_row + i, b_tile_col, BN)]) = FETCH_FLOAT4(B[OFFSET(b_tile_row + i, b_tile_col, N)]);
    }

    __syncthreads();
#pragma unroll
    for (int m = 0; m < TM; m += 4){
        FETCH_FLOAT4(a_frag[0][m]) = FETCH_FLOAT4(As[0][OFFSET(0, ty + m, BM)]);
    }
#pragma unroll
    for (int n = 0; n < TN; n += 4){
        FETCH_FLOAT4(b_frag[0][n]) = FETCH_FLOAT4(Bs[0][OFFSET(0, tx + n, BN)]);
    }


    int write_index = 1;
    int load_index;
    int k = 0;
    do{
        k += BK;
        if (k < K){
            // 预取
#pragma unroll            
            for (int i = 0; i < BM; i += a_tile_stride){
                int ldg_index = i / a_tile_stride * 4;
                // 提前读入下一块数据( A 的下一列)
                FETCH_FLOAT4(ldg_a_reg[ldg_index]) = FETCH_FLOAT4(A[OFFSET(a_tile_row + i, a_tile_col + k, K)]);
            }
#pragma unroll
            for (int i = 0; i < BK; i += b_tile_stride){
                int ldg_index = i / b_tile_stride * 4;
                //  提前读入下一块数据( B 的下一行)
                FETCH_FLOAT4(ldg_b_reg[ldg_index]) = FETCH_FLOAT4(B[OFFSET(b_tile_row + i + k, b_tile_col, N)]);
            }
        }   // if (k < K)

        load_index = write_index ^ 1;
#pragma unroll
        for (int bk = 0; bk < BK - 1; bk++){
#pragma unroll           
            for (int m = 0; m < TM; m += 4){
                FETCH_FLOAT4(a_frag[(bk + 1) % 2][m]) = FETCH_FLOAT4(As[load_index][OFFSET(bk + 1, ty + m, BM)]);
            }   // 提前把下一块装进来
#pragma unroll
            for (int n = 0; n < TN; n += 4){
                FETCH_FLOAT4(b_frag[(bk + 1) % 2][n]) = FETCH_FLOAT4(Bs[load_index][OFFSET(bk + 1, tx + n, BN)]);
            }

#pragma unroll            
            for (int m = 0; m < TM; m++){
                for (int n = 0; n < TN; n++){
                    accum[m][n] += a_frag[bk % 2][m] * b_frag[bk % 2][n];
                }
            }   // 计算当前块, bk = 0 时, a_frag 和 b_frag 已经装入了数据
        }   // 当前 tile 的计算            
        
        

        if (k < K){
#pragma unroll            
            for (int i = 0; i < BM; i += a_tile_stride){
                int ldg_index = i / a_tile_stride * 4;
                As[write_index][OFFSET(a_tile_col, i + a_tile_row, BM)] = ldg_a_reg[ldg_index];
                As[write_index][OFFSET(a_tile_col + 1, i + a_tile_row, BM)] = ldg_a_reg[ldg_index + 1];
                As[write_index][OFFSET(a_tile_col + 2, i + a_tile_row, BM)] = ldg_a_reg[ldg_index + 2];
                As[write_index][OFFSET(a_tile_col + 3, i + a_tile_row, BM)] = ldg_a_reg[ldg_index + 3];
            }   // 从 A 预取数据
#pragma unroll
            for (int i = 0; i < BK; i += b_tile_stride){
                int ldg_index = i / b_tile_stride * 4;
                FETCH_FLOAT4(Bs[write_index][OFFSET(i + b_tile_row, b_tile_col, BN)]) =  FETCH_FLOAT4(ldg_b_reg[ldg_index]);
            }   // 从 B 预取数据


            __syncthreads();
#pragma unroll
            // 把下一块 tile 的第 0 行放进来
            for (int m = 0; m < TM; m += 4){
                FETCH_FLOAT4(a_frag[0][m]) = FETCH_FLOAT4(As[write_index][OFFSET(0, ty + m, BM)]);
            }
#pragma unroll
            for (int n = 0; n < TN; n += 4){
                FETCH_FLOAT4(b_frag[0][n]) = FETCH_FLOAT4(Bs[write_index][OFFSET(0, tx + n, BN)]);
            }

            write_index ^= 1;            
        }   // if (k < K)
      
#pragma unroll
        for (int m = 0; m < TM; m++){
#pragma unroll            
            for (int n = 0; n < TN; n++){
                accum[m][n] += a_frag[(BK - 1) % 2][m] * b_frag[(BK - 1) % 2][n];
            }
        }

    }while(k < K);

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


void checkCudaError(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        std::cerr << msg << " CUDA ERROR: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

void checkCublasError(cublasStatus_t status, const char *msg) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << msg << " CUBLAS ERROR: " << status << std::endl;
        exit(EXIT_FAILURE);
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
        checkCudaError(cudaMalloc(&d_A, numBytes), "cudaMalloc d_A failed");
        checkCudaError(cudaMalloc(&d_B, numBytes), "cudaMalloc d_B failed");
        checkCudaError(cudaMalloc(&d_C_v1, numBytes), "cudaMalloc d_C_v1 failed");

        try{
            for (int i = 0; i < elemCount; i++){
                A[i] = 1.0f;
                B[i] = 2.0f;
            }

            checkCudaError(cudaMemcpy(d_A, A, numBytes, cudaMemcpyHostToDevice), "cudaMemcpy A to device failed");
            checkCudaError(cudaMemcpy(d_B, B, numBytes, cudaMemcpyHostToDevice), "cudaMemcpy A to device failed");
            
            cublasHandle_t handle;  //定义句柄, 用于记录状态
            checkCublasError(cublasCreate(&handle), "cublasCreate failed");  // 创建cuBLAS上下文

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

            /*------------------------v3计算------------------------*/
            dim3 threads(256); // 
            dim3 blocks((N + 128 - 1) / 128, (N + 128 - 1) / 128);
            
            for (int i = 0; i < warmup_time; i++){
                // BM, BN, BK, TM, TN
                mysgemm_v5<128, 128, 8, 8, 8><<<blocks, threads>>>(N, N, N, alpha, d_A, d_B, beta, d_C_v1);
            }
            cudaDeviceSynchronize();

            cudaEventRecord(start);
            for (int i = 0; i < repeat_time; i++){
                // BM, BN, BK, TM, TN
                mysgemm_v5<128, 128, 8, 8, 8><<<blocks, threads>>>(N, N, N, alpha, d_A, d_B, beta, d_C_v1);
            }
            cudaEventRecord(end);
            cudaEventSynchronize(end);

            float v4_time = 0.f;
            cudaEventElapsedTime(&v4_time, start, end);

            cudaMemcpy(C_v1, d_C_v1, numBytes, cudaMemcpyDeviceToHost);
            std::cout<<"v4 time:"<<v4_time<<"ms"<<std::endl;

            float cublas_gflops = 2.f * N * N * N * repeat_time / (cublas_time * 1e6f);
            float v4_gflops = 2.f * N * N * N * repeat_time / (v4_time * 1e6f);

            std::cout<< "cublas GFLOPS:" << cublas_gflops << std::endl;
            std::cout<< "v4 GFLOPS:" << v4_gflops << std::endl;
            std::cout<< "precent:" << (v4_gflops / cublas_gflops) * 100 << "%" << std::endl;

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