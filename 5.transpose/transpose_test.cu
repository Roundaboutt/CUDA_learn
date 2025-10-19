#include <iostream>
#include "cuda_runtime.h"

float* transpose_cpu(float* output, int nx, int ny){

    float* input = (float*)malloc(sizeof(float) * nx * ny);
    for (int i = 0; i < nx * ny; i++){
        input[i] = float(i);
    }

    for (int i = 0; i < ny; i++){
        for (int j = 0; j < nx; j++){
            // output[j][i] = input[i][j];
            output[j * ny + i] = input[i * nx + j];
        }
    }
    return output;
}

#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*> (&(pointer))[0])
#define OFFSET(row, col, ld) ((row) * (ld) + col)
// tile + shared memory优化
template <const int BM, const int BN, const int TM, const int TN>
__global__ void transpose_test(float* input, float* output, int M, int N){
    __shared__ float tile[BM][BN];

    // block的坐标
    int block_row = blockIdx.y * BM;
    int block_col = blockIdx.x * BN;

    // 线程在block内的坐标
    int thread_row = threadIdx.y * TM;
    int thread_col = threadIdx.x * TN;

    for (int m = 0; m < TM ; m++){
        int global_row = block_row + thread_row + m;
        for (int n = 0; n < TN; n += 4){
            int global_col = block_col + thread_col + n;
            if (global_row < M && global_col < N){
                FETCH_FLOAT4(tile[thread_row + m][thread_col + n]) = FETCH_FLOAT4(input[OFFSET(global_row, global_col, N)]);
            }
        }
    }

    __syncthreads();

    int trans_block_row = blockIdx.x * BN;
    int trans_block_col = blockIdx.y * BM;

    int trans_thread_row = threadIdx.x * TN;
    int trans_thread_col = threadIdx.y * TM;

    float temp[TN];

    for (int m = 0; m < TM; m++){
        int global_col = trans_block_col + trans_thread_col + m;
        for (int n = 0; n < TN; n += 4){
            int global_row = trans_block_row + trans_thread_row + n;
            if (global_col < M && global_row < N){
                FETCH_FLOAT4(temp) = FETCH_FLOAT4(tile[thread_row + m][thread_col + n]);   
                
                output[OFFSET(trans_block_row, trans_block_col, M)] = temp[0];
                output[OFFSET(trans_block_row + 1, trans_block_col, M)] = temp[1];
                output[OFFSET(trans_block_row + 2, trans_block_col, M)] = temp[2];
                output[OFFSET(trans_block_row + 3, trans_block_col, M)] = temp[3];
            }

        }
    }
}


void call_test(float* input, float* output, int M, int N){
    const int BM = 64, BN = 64, TM = 4, TN = 4;
    dim3 blocksize(BN / TN, BM / TM);
    dim3 gridsize((N + blocksize.x - 1) / blocksize.x, (M + blocksize.y - 1) / blocksize.y);

    transpose_test<BM, BN, TM, TN><<<gridsize, blocksize>>>(input, output, M, N);
}

bool isMatch(float* a, float* b, int elemCount){
    for (int i = 0; i < elemCount; i++){
        if (fabsf(a[i] - b[i]) > 1e-5) return false;
    }
    return true;
}

int main(){
    int M = 1024, N = 1024;
    int numBytes = M * N * sizeof(float);
    float* h_input = (float*)malloc(numBytes);
    float* h_output_cpu = (float*)malloc(numBytes);
    float* h_output_gpu = (float*)malloc(numBytes);

    for (int i = 0; i < M * N; i++){
        h_input[i] = float(i);
    }

    float* d_input, * d_output;
    cudaMalloc((void**)&d_input, numBytes);
    cudaMalloc((void**)&d_output, numBytes);

    cudaMemcpy(d_input, h_input, numBytes, cudaMemcpyHostToDevice);

    call_test(d_input, d_output, M, N);

    cudaMemcpy(h_output_gpu, d_output, numBytes, cudaMemcpyDeviceToHost);

    h_output_cpu = transpose_cpu(h_output_cpu, M, N);

    if (isMatch(h_output_cpu, h_output_gpu, M * N)){
        std::cout << "match!" << std::endl;
    }
    else{
        std::cout << "not match!" << std::endl;
    }


}