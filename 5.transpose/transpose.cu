#include <iostream>
#include "cuda_runtime.h"

__global__ void transpose_v1(float* output, float* input, int nx, int ny){
    int id_x = threadIdx.x + blockIdx.x * blockDim.x;
    int id_y = threadIdx.y + blockIdx.y * blockDim.y;

    if (id_x >= nx || id_y >= ny) return;
    output[id_x * ny + id_y] = input[id_y * nx + id_x];
}

void call_v1(float* d_output, float* d_input, int nx, int ny){
    dim3 blockSize(16, 16);
    dim3 gridSize((nx + blockSize.x - 1) / blockSize.x, (ny + blockSize.y - 1) / blockSize.y);
    transpose_v1<<<gridSize, blockSize>>>(d_output, d_input, nx, ny);
}


template <const int SUBX, const int SUBY>
__global__ void transpose_v2(float* output, float* input, int nx, int ny){

    __shared__ float tile[SUBY][SUBX];

    int id_x = threadIdx.x + blockIdx.x * blockDim.x;
    int id_y = threadIdx.y + blockIdx.y * blockDim.y;   
    int thread_in = id_y * nx + id_x;

    // 一个block内的二维坐标转化为一维坐标
    int bidx = threadIdx.y * blockDim.x + threadIdx.x;
    // 把这个一维的坐标重排成转置后的二维坐标
    int idx_row = bidx / blockDim.y;
    int idx_col = bidx % blockDim.y;

    // 转置后矩阵的坐标
    id_x = idx_col + blockIdx.y * blockDim.y;
    id_y = idx_row + blockIdx.x * blockDim.x;

    int thread_out = id_y * ny + id_x;

    if (id_x < nx && id_y < ny){
        tile[threadIdx.y][threadIdx.x] = input[thread_in];
        
        __syncthreads();

        output[thread_out] = tile[idx_col][idx_row];
    }
}

void call_v2(float* d_output, float* d_input, int nx, int ny){
    dim3 blockSize(16, 16);
    dim3 gridSize((nx + blockSize.x - 1) / blockSize.x, (ny + blockSize.y) / blockSize.y);

    transpose_v2<32, 16><<<gridSize, blockSize>>>(d_output, d_input, nx, ny);

}



float* v1_time(int nx, int ny){

    int elemCount = nx * ny;
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

    int warmup = 10;
    for (int i = 0; i < warmup; i++){
        call_v1(d_output, d_input, nx, ny);        
    }

    int repeat_time = 10;

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    cudaEventRecord(start);
    for (int i = 0; i < repeat_time; i++){
        call_v1(d_output, d_input, nx, ny);
    }
    cudaEventRecord(end);
    cudaEventSynchronize(end);

    float v1_time = 0;
    cudaEventElapsedTime(&v1_time, start, end);
    std::cout << "v1 time:" << v1_time << "ms" << std::endl;

    cudaDeviceSynchronize();
    cudaMemcpy(h_output, d_output, numBytes, cudaMemcpyDeviceToHost);

    cudaEventDestroy(start);
    cudaEventDestroy(end);

    cudaFree(d_output);
    cudaFree(d_input);
    free(h_input);
    return h_output;  
}

float* v2_time(int nx, int ny){
    
    int elemCount = nx * ny;
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

    int warmup = 10;
    for (int i = 0; i < warmup; i++){
        call_v2(d_output, d_input, nx, ny);        
    }

    int repeat_time = 10;

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    cudaEventRecord(start);
    for (int i = 0; i < repeat_time; i++){
        call_v2(d_output, d_input, nx, ny);
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
    int nx = 4096, ny = 4096;
    int numBytes = nx * ny * sizeof(float);
    float* v1_output = (float*)malloc(numBytes);
    float* v2_output = (float*)malloc(numBytes);
    v1_output = v1_time(nx, ny);
    v2_output = v2_time(nx, ny);

    if (isMatch(v1_output, v2_output, nx * ny)){
        std::cout << "Results Match!" << std::endl;
    }
}

