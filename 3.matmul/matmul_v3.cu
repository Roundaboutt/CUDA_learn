#include<iostream>
#include<cuda_runtime.h>
#include<vector>
#include<cublas_v2.h>


#define OFFSET(row, col, ld) ((row) * (ld) + (col))
template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void sgemm_v3(float* A, float* B, float* C, int M, int N, int K, float alpha, float beta){
    const int tx = threadIdx.x * TN;
    const int ty = threadIdx.y * TM;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;

    const int block_row_thread = BN / TN;
    const int block_col_thread = BM / TM;
    const int num_thread = block_row_thread * block_col_thread;

    const int linead_tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int a_tile_row = linead_tid / BK;
    const int a_tile_col = linead_tid % BK;
    const int a_tile_stride = num_thread / BK;

    const int b_tile_row = linead_tid / BN;
    const int b_tile_col = linead_tid % BN;
    const int b_tile_stride = num_thread / BN;

    float temp[TM][TN] = {0.f};

    A = &A[by * K * BM];
    B = &B[bx * BN];
    C = &C[by * BM * N + bx * BN];

    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];


#pragma unroll
    for (int k = 0; k < K; k += BK)
    {
#pragma unroll
        for (int i = 0; i < BM; i += a_tile_stride)
        {
            As[a_tile_row + i][a_tile_col] = A[OFFSET(a_tile_row + i, a_tile_col, K)];
        }
#pragma unroll
        for (int i = 0; i < BK; i += b_tile_stride)
        {
            Bs[b_tile_row + i][b_tile_col] = B[OFFSET(b_tile_row + i, b_tile_col, N)];
        }        

        __syncthreads();

        A += BK;
        B += BK * N;

#pragma unroll
        for (int k1 = 0; k1 < BK; k1++)
        {
#pragma unroll
            for (int m = 0; m < TM; m++)
            {
#pragma unroll
                for (int n = 0; n < TN; n++)
                {
                    temp[m][n] += As[ty + m][k1] * Bs[k1][tx + n];
                }                
            }            
        }

        __syncthreads();
    }    

#pragma unroll
    for (int m = 0; m < TM; m++)
    {
#pragma unroll
        for (int n = 0; n < TN; n++)
        {
            C[OFFSET(ty + m, tx + n, N)] = alpha * temp[m][n] + beta * C[OFFSET(ty + m, tx + n, N)];
        }        
    }    
}

void call_v3(float* A, float* B, float* C, int M, int N, int K){
    const int alpha = 1;
    const int beta = 0;

    const int BM = 128;
    const int BN = 128;
    const int BK = 8;
    const int TM = 8;
    const int TN = 8;


    dim3 blocksize(BN / TN, BM / TM);
    dim3 gridsize((N + BN - 1) / BN, (M + BM - 1) / BM);

    sgemm_v3<BM, BN, BK, TM, TN><<<gridsize, blocksize>>>(A, B, C, M, N, K, alpha, beta);
    
}

void init_data(float* A, int row, int col){
    for (int i = 0; i < row; i++){
        for (int j = 0; j < col; j++){
            A[OFFSET(i, j, col)] = float(std::rand() % 5);
        }
    }
}

// helper: row-major (rows x cols) -> column-major (rows x cols)
void row_to_col_major(const float* src_row, float* dst_col, int rows, int cols){
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            dst_col[j*rows + i] = src_row[i*cols + j];
}

float* cublas_time(float* h_A, float* h_B, float* h_C, int M, int N, int K){
    int size_A = M * K; // A: M x K (row-major)
    int size_B = K * N; // B: K x N (row-major)
    int size_C = M * N; // C: M x N

    // prepare column-major copies on host
    float* h_A_col = (float*)malloc(sizeof(float) * size_A);
    float* h_B_col = (float*)malloc(sizeof(float) * size_B);
    float* h_C_col = (float*)malloc(sizeof(float) * size_C);

    row_to_col_major(h_A, h_A_col, M, K); // A_row -> A_col (size M*K stored col-major)
    row_to_col_major(h_B, h_B_col, K, N); // B_row -> B_col (size K*N stored col-major)
    // init h_C_col to zeros
    memset(h_C_col, 0, sizeof(float) * size_C);

    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, sizeof(float) * size_A);
    cudaMalloc((void**)&d_B, sizeof(float) * size_B);
    cudaMalloc((void**)&d_C, sizeof(float) * size_C);

    // copy column-major buffers to device
    cudaMemcpy(d_A, h_A_col, sizeof(float) * size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B_col, sizeof(float) * size_B, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C_col, sizeof(float) * size_C, cudaMemcpyHostToDevice);

    cublasHandle_t handle;
    cublasCreate(&handle);
    float alpha = 1.f, beta = 0.f;

    // NOTE: Now use (M, N, K) and leading dims for column-major:
    // lda = M (rows of A), ldb = K (rows of B), ldc = M (rows of C).
    int warmup_time = 10;
    for (int i = 0; i < warmup_time; ++i){
        cublasSgemm(handle,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    M, N, K,
                    &alpha,
                    d_A, M,
                    d_B, K,
                    &beta,
                    d_C, M);
    }
    cudaDeviceSynchronize();

    int repeat_time = 10;
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);
    for (int i = 0; i < repeat_time; ++i){
        cublasSgemm(handle,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    M, N, K,
                    &alpha,
                    d_A, M,
                    d_B, K,
                    &beta,
                    d_C, M);
    }
    cudaEventRecord(end);
    cudaEventSynchronize(end);

    float cublas_time_ms = 0;
    cudaEventElapsedTime(&cublas_time_ms, start, end);
    std::cout << "cublas time: " << cublas_time_ms << " ms" << std::endl;

    // copy result back (column-major) then convert to row-major for comparison
    cudaMemcpy(h_C_col, d_C, sizeof(float) * size_C, cudaMemcpyDeviceToHost);
    // column-major -> row-major: h_C[row * N + col] = h_C_col[col*M + row]
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j)
            h_C[OFFSET(i, j, N)] = h_C_col[j * M + i];

    // cleanup
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A_col); free(h_B_col); free(h_C_col);
    cublasDestroy(handle);
    cudaEventDestroy(start); cudaEventDestroy(end);

    return h_C;
}


float* v3_time(float* h_A, float* h_B, float* h_C, int M, int N, int K){
    int size_A = M * K;
    int size_B = N * K;
    int size_C = M * N;

    float* d_A,* d_B,* d_C;
    cudaMalloc((void**)&d_A, sizeof(float) * size_A);
    cudaMalloc((void**)&d_B, sizeof(float) * size_B);
    cudaMalloc((void**)&d_C, sizeof(float) * size_C);

    cudaMemcpy(d_A, h_A, sizeof(float) * size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeof(float) * size_B, cudaMemcpyHostToDevice);

    int warmup_time = 10;
    int repeat_time = 10;
    for (int i = 0; i < warmup_time; i++){
        call_v3(d_A, d_B, d_C, M, N, K);
    }

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    cudaEventRecord(start);
    for (int i = 0; i < repeat_time; i++){
        call_v3(d_A, d_B, d_C, M, N, K);
    }
    cudaEventRecord(end);
    cudaEventSynchronize(end);

    float elapse_time = 0;
    cudaEventElapsedTime(&elapse_time, start, end);

    std::cout << "v3 time: " << elapse_time << "ms" << std::endl;

    cudaMemcpy(h_C, d_C, sizeof(float) * size_C, cudaMemcpyDeviceToHost);


    cudaEventDestroy(start);
    cudaEventDestroy(end);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return h_C;
}


bool isMatch(float* a, float* b, int elemCount){
    for (int i = 0; i < elemCount; i++){
        if (fabsf(a[i] - b[i]) > 1e-5) return false;
    }
    return true;
}

int main(){

    std::vector<int> sizes = {128, 256, 512, 1024, 2048, 4096, 8192};

    float* h_A,* h_B,* h_C_sgemm,* h_C_cublas;
    for (int size:sizes){
        std::cout << "-----Test Size:"  << size << "-----" << std::endl;
        int M = size, N = size, K = size;
        int size_A = M * K;
        int size_B = N * K;
        int size_C = M * N;

        h_A = (float*)malloc(size_A * sizeof(float));
        h_B = (float*)malloc(size_B * sizeof(float));
        h_C_sgemm = (float*)malloc(size_C * sizeof(float));
        h_C_cublas = (float*)malloc(size_C * sizeof(float));

        init_data(h_A, M, K);
        init_data(h_B, K, N);

        h_C_sgemm = v3_time(h_A, h_B, h_C_sgemm, M, N, K);
        h_C_cublas = cublas_time(h_A, h_B, h_C_cublas, M, N, K);

        if (isMatch(h_C_sgemm, h_C_cublas, size_C)){
            std::cout << "Results Match!" << std::endl;
        }
        else{
            std::cout << "Results not Match!" << std::endl;
        }  

    }

}
