#include <iostream>
#include "cuda_runtime.h"

float* transpose_cpu(float* output, int nx, int ny){

    float* input =(float*) malloc(sizeof(float) * nx * ny);
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


__global__ void transpose_v1(float* output, float* input, int nx, int ny){
    const int id_x = threadIdx.x + blockDim.x * blockIdx.x;
    const int id_y = threadIdx.y + blockDim.y * blockIdx.y;

    if (id_x >= nx || id_y >= ny) return;
    output[id_x * ny + id_y] = input[id_y * nx + id_x];
}

void call_v1(float* d_output, float* d_input, int nx, int ny, int BLKDIM_x, int BLKDIM_y){
    dim3 blockSize(BLKDIM_x, BLKDIM_y);
    dim3 gridSize((nx + blockSize.x - 1) / blockSize.x, (ny + blockSize.y - 1) / blockSize.y);
    transpose_v1<<<gridSize, blockSize>>>(d_output, d_input, nx, ny);
}

float* v1_time(int nx, int ny, int warmup, int repeat_time, int BLKDIM_x, int BLKDIM_y){

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

    for (int i = 0; i < warmup; i++){
        call_v1(d_output, d_input, nx, ny, BLKDIM_x, BLKDIM_y);        
    }


    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    cudaEventRecord(start);
    for (int i = 0; i < repeat_time; i++){
        call_v1(d_output, d_input, nx, ny, BLKDIM_x, BLKDIM_y);
    }
    cudaEventRecord(end);
    cudaEventSynchronize(end);

    float v1_time = 0;
    cudaEventElapsedTime(&v1_time, start, end);
    std::cout << "blockdim:(" << BLKDIM_x << "," << BLKDIM_y << ")" << std::endl;
    std::cout << "v1 time:" << v1_time << "ms" << std::endl;
    std::cout << "---------------------" << std::endl;
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

    float* cpu_output = (float*)malloc(numBytes);    

    cpu_output = transpose_cpu(cpu_output, nx, ny);

    float* v1_output1 = v1_time(nx, ny, 10, 10, 8, 32);
    float* v1_output2 = v1_time(nx, ny, 10, 10, 32, 8);

    if (isMatch(cpu_output, v1_output1, nx * ny)){
        std::cout << "Results1 Match!" << std::endl;
    }
    else{
        std::cout << "Results1 not Match!" << std::endl;
    }

    if (isMatch(cpu_output, v1_output2, nx * ny)){
        std::cout << "Results2 Match!" << std::endl;
    }
    else{
        std::cout << "Results2 not Match!" << std::endl;
    }
}
