#include<iostream>
#include<chrono>
#include"cuda_runtime.h"
#include<cublas_v2.h>
#include<vector>


#define OFFSET(row, col ,ld) ((row) * (ld) + (col))
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

template <const int BM, const int BN, const int BK, const int row_stride_a, const int row_stride_b>
__device__ void load_sgemm(int N, int K, float* A, float* B, float* As, float* Bs, 
int inner_row_a, int inner_col_a, int inner_row_b, int inner_col_b){
    for (int i = 0; i + row_stride_a <= BM; i += row_stride_a){
        float4 temp = FETCH_FLOAT4(A[OFFSET(i + inner_row_a, inner_col_a * 4, K)]);
        As[OFFSET(inner_col_a * 4, inner_row_a + i, BM)] = temp.x;
        As[OFFSET(inner_col_a * 4 + 1, inner_row_a + i, BM)] = temp.y;
        As[OFFSET(inner_col_a * 4 + 2, inner_row_a + i, BM)] = temp.z;
        As[OFFSET(inner_col_a * 4 + 3, inner_row_a + i, BM)] = temp.w;
    }

    for (int i = 0; i + row_stride_b <= BK; i += row_stride_b){
        FETCH_FLOAT4(Bs[OFFSET(i + inner_row_b, inner_col_b * 4, BN)]) = FETCH_FLOAT4(B[OFFSET(i + inner_row_b, inner_col_b * 4, N)]);
    }
}


template <const int BM, const int BN, const int BK, const int WM, const int WN, const int WMITER, const int WNITER, 
const int WSUBM, const int WSUBN, const int TM, const int TN>
__device__ void compute_sgemm(float* reg_m, float* reg_n, float* thread_results, const float* As, const float* Bs, 
const int warp_row, const int warp_col, const int thread_row_inwarp, const int thread_col_inwarp){
    
    for (int k = 0; k < BK; k++){
        // 搬运As的行到reg_m
        for (int w_sub_row_idx = 0; w_sub_row_idx < WMITER; w_sub_row_idx++){
            for (int m = 0; m < TM; m++){
                reg_m[w_sub_row_idx * TM + m] = As[k * BM + warp_row * WM + w_sub_row_idx * WSUBM + thread_row_inwarp * TM + m];
            }
        }


        // 搬运Bs的行到reg_n
        for (int w_sub_col_idx = 0; w_sub_col_idx < WNITER; w_sub_col_idx++){
            for (int n = 0; n < TN; n++){
                reg_n[w_sub_col_idx * TN + n] = Bs[k * BN + warp_col * WN + w_sub_col_idx * WSUBN + thread_col_inwarp * TN + n];
            }
        }

        for (int w_sub_row_idx = 0; w_sub_row_idx < WMITER; w_sub_row_idx++){
            for (int w_sub_col_idx = 0; w_sub_col_idx < WNITER; w_sub_col_idx++){
                for (int m = 0; m < TM; m++){
                    for (int n = 0; n < TN; n++){
                        thread_results[(w_sub_row_idx * WM + m) * (WNITER * TN) + w_sub_col_idx * TN + n] += 
                        reg_m[w_sub_row_idx * TM + m] * reg_n[w_sub_col_idx * TN + n];
                    }
                }
            }
        }
    }
}   


constexpr int WARP_SIZE = 32;
template <const int BM, const int BN, const int BK, const int WM, const int WN,
 const int WNITER, const int TM, const int TN, const int NUM_THREADS>
__global__ void sgemm_v6(int M, int N, int K, float* A, float* B, float* C, float alpha, float beta){
    const int by = blockIdx.y;
    const int bx = blockIdx.x;

    const int warp_idx = threadIdx.x / WARP_SIZE;   // 当前线程所属warp在线程块内的编号
    // 将一维的warp_idx映射成二维的坐标(warp_row, warp_col)
    const int warp_col = warp_idx % (BN / WN);  // 0 <= warp_col <= BN/WN
    const int warp_row = warp_idx / (BN / WN);  // 0 <= warp_row <= BM/WM

    constexpr int WMITER = (WM * WN) / (WARP_SIZE * TM * TN * WNITER);
    constexpr int WSUBM = WM / WMITER;
    constexpr int WSUBN = WN / WNITER;



    // 当前线程在 WSUBMxWSUBN 范围内的局部坐标
    const int thread_idx_inwarp = threadIdx.x % WARP_SIZE;
    const int thread_row_inwarp = thread_idx_inwarp / (WSUBN / TN);
    const int thread_col_inwarp = thread_idx_inwarp % (WSUBN / TN);


    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    A = &A[by * BM * K];
    B = &B[bx * BN];
    C = &C[(by * BM + warp_row * WM) * N + bx * BN + warp_col * WN];

    const int inner_row_a = threadIdx.x / (BK / 4);
    const int inner_col_a = threadIdx.x % (BK / 4);
    const int row_stride_a = (NUM_THREADS * 4) / BK;

    const int inner_row_b = threadIdx.x / (BN / 4);
    const int inner_col_b = threadIdx.x % (BN / 4);
    const int row_stride_b = (NUM_THREADS * 4) / BN;

    float thread_results[WMITER * TM * WNITER * TN] = {0.};
    float reg_m[WMITER * TM] = {0.};
    float reg_n[WNITER * TN] = {0.};

    for (int bk = 0; bk < K; bk += BK){
        load_sgemm<BM, BN, BK, row_stride_a, row_stride_b>(N, K, A, B, As, Bs, inner_row_a, inner_col_a, inner_row_b, inner_col_b);

        __syncthreads();

        compute_sgemm<BM, BN, BK, WM, WN, WMITER, WNITER, WSUBM, WSUBN, TM, TN>(reg_m, reg_n, thread_results, 
            As, Bs, warp_row, warp_col, thread_row_inwarp, thread_col_inwarp);

        A += BK;
        B += BK * N;
        __syncthreads();
    }


    
    for (int w_sub_row_idx = 0; w_sub_row_idx < WMITER; w_sub_row_idx++){
        for (int w_sub_col_idx = 0; w_sub_col_idx < WNITER; w_sub_col_idx++){
            // 遍历所有的(WMITER * TM) x (WNITER * TN)块
            float* C_interim = C + (w_sub_row_idx * WSUBM) * N + w_sub_col_idx * WSUBN;
            

            // 遍历每个线程负责的元素
            for (int m = 0; m < TM; m++){
                for (int n = 0; n < TN; n += 4){
                    float4 temp = FETCH_FLOAT4(C_interim[(thread_row_inwarp * TM + m) * N + thread_col_inwarp * TN + n]);
                    const int i = (w_sub_row_idx * TM + m) * (WNITER * TN) + w_sub_col_idx * TN + n;
                    temp.x = alpha * thread_results[i] + beta * temp.x;
                    temp.y = alpha * thread_results[i + 1] + beta * temp.y;
                    temp.z = alpha * thread_results[i + 2] + beta * temp.z;
                    temp.w = alpha * thread_results[i + 3] + beta * temp.w;

                    FETCH_FLOAT4(C_interim[(thread_row_inwarp * TM + m) * N + thread_col_inwarp * TN + n]) = temp;
                }
            }
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

            int warmup_time = 10;
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

            /*------------------------v6计算------------------------*/

            const int NUM_THREADS = 128;
            const int BN = 128;
            const int BM = 128;
            const int BK = 8;
            const int WN = 64;
            const int WM = 64;
            const int WNITER = 4;
            const int TN = 4;
            const int TM = 8;


            dim3 threads(NUM_THREADS);
            dim3 blocks((N + NUM_THREADS - 1) / NUM_THREADS, (N + NUM_THREADS - 1) / NUM_THREADS);
            
            for (int i = 0; i < warmup_time; i++){
                sgemm_v6<BM, BN, BK, WM, WN, WNITER, TM, TN, NUM_THREADS><<<blocks, threads>>>(N, N, N, d_A, d_B, d_C_sgemm, alpha, beta);
            }
            cudaDeviceSynchronize();

            cudaEventRecord(start);
            for (int i = 0; i < repeat_time; i++){
                sgemm_v6<BM, BN, BK, WM, WN, WNITER, TM, TN, NUM_THREADS><<<blocks, threads>>>(N, N, N, d_A, d_B, d_C_sgemm, alpha, beta);
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