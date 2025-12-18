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


template <const int BLKDIM_X, const int BLKDIM_Y, const int PAD>
__global__ void transpose_v2(float* output, float* input, int M, int N) {

    __shared__ float tile[BLKDIM_Y][BLKDIM_X + PAD];

    const int id_x = threadIdx.x + blockIdx.x * blockDim.x;
    const int id_y = threadIdx.y + blockIdx.y * blockDim.y;

    if (id_x < M && id_y < N)
    {
        tile[threadIdx.y][threadIdx.x] = input[id_y * M + id_x];
    }

    __syncthreads();

    const int block_x_out = blockIdx.y * BLKDIM_Y;
    const int block_y_out = blockIdx.x * BLKDIM_X;

    const int trans_x = block_x_out + threadIdx.x;
    const int trans_y = block_y_out + threadIdx.y;

    if (trans_x < N && trans_y < M)
    {
        output[trans_y * N + trans_x] = tile[threadIdx.x][threadIdx.y];
    }
}

template <const int BLKDIM_X, const int BLKDIM_Y, const int PAD>
void call_v2(float* d_output, float* d_input, int M, int N){
    dim3 blockSize(BLKDIM_X, BLKDIM_Y);
    dim3 gridSize((M + blockSize.x - 1) / blockSize.x, (N + blockSize.y) / blockSize.y);

    transpose_v2<BLKDIM_X, BLKDIM_Y, PAD><<<gridSize, blockSize>>>(d_output, d_input, M, N);

}

template <const int BLKDIM_X, const int BLKDIM_Y, const int PAD>
float* v2_time(int M, int N, int warmup, int repeat_time){
    
    int elemCount = M * N;
    int numBytes = elemCount * sizeof(float);

    float* h_input,* h_output;
    h_input = (float*)malloc(numBytes);
    h_output = (float*)malloc(numBytes);

    for (int i = 0; i < elemCount; i++){
        h_input[i] = float(i);
    }


    float* d_input,* d_output;
    cudaMalloc((void**)&d_input, numBytes);
    cudaMalloc((void**)&d_output, numBytes);

    cudaMemcpy(d_input, h_input, numBytes, cudaMemcpyHostToDevice);

    for (int i = 0; i < warmup; i++){
        call_v2<BLKDIM_X, BLKDIM_Y, PAD>(d_output, d_input, M, N);        
    }


    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    cudaEventRecord(start);
    for (int i = 0; i < repeat_time; i++){
        call_v2<BLKDIM_X, BLKDIM_Y, PAD>(d_output, d_input, M, N);
    }
    cudaEventRecord(end);
    cudaEventSynchronize(end);

    float v2_time = 0;
    cudaEventElapsedTime(&v2_time, start, end);
    std::cout << "v2 time:" << v2_time << "ms" << std::endl;

    cudaDeviceSynchronize();
    cudaMemcpy(h_output, d_output, numBytes, cudaMemcpyDeviceToHost);

    cudaEventDestroy(start);
    cudaEventDestroy(end);

    cudaFree(d_output);
    cudaFree(d_input);
    free(h_input);
    return h_output;
}

bool isMatch(float* a, float* b, int elemCount){
    for (int i = 0; i < elemCount; i++){
        if (fabsf(a[i] - b[i]) > 1e-5) return false;
    }
    return true;
}

int main(){
    int M = 512, N = 2048;
    int numBytes = M * N * sizeof(float);


    float* cpu_output = (float*)malloc(numBytes);    
    float* v1_output = (float*)malloc(numBytes);
    float* v2_output = (float*)malloc(numBytes);

    cpu_output = transpose_cpu(cpu_output, M, N);
    std::cout << "PAD:" << 0 << std::endl;
    v2_output = v2_time<32, 32, 0>(M, N, 10, 10);
    std::cout << "PAD:" << 1 << std::endl;
    v2_output = v2_time<32, 32, 1>(M, N, 10, 10);

    if (isMatch(cpu_output, v2_output, M * N)){
        std::cout << "Results Match!" << std::endl;
    }
    else{
        std::cout << "Results not Match!" << std::endl;
    }
}

