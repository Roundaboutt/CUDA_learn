#include<iostream>
#include<chrono>
#include"cuda_runtime.h"

#define N 1024 * 1024
#define BLOCKSIZE 1024

__global__ void reduce_v2(float* input, float* output){
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int id = tid + bid * blockDim.x * 2;

    __shared__ float shared[BLOCKSIZE];
    shared[tid] = input[id] + input[id + blockDim.x];

    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1){
        if (tid < s){
            shared[tid] += shared[tid + s];            
        }
        __syncthreads();
    }

    if (tid == 0) output[bid] = shared[0];
}

float reduce_cpu(float* input){
    float sum = 0.0f;
    for (int i = 0;i < N; i++){
        sum += input[i];
    }
    return sum;
}

int main(){
    const int num_blocks = ((N + BLOCKSIZE - 1) / BLOCKSIZE) / 2;

    const int elemCount = N;
    const int numBytes = N * sizeof(float);

    float* h_input = (float*)malloc(numBytes);

    for (int i = 0;i < elemCount; i++){
        h_input[i] = 1.0f;
    }
    
    auto cpu_start = std::chrono::high_resolution_clock::now();
    float cpu_res = reduce_cpu(h_input);
    auto cpu_end = std::chrono::high_resolution_clock::now();    
    std::chrono::duration<double, std::milli> cpu_time = cpu_end - cpu_start;

    std::cout<<"cpu result:"<<cpu_res<<std::endl;
    std::cout<<"cpu time:"<<cpu_time.count()<<"ms"<<std::endl;

    float* d_input,* d_output,* d_final_res;
    float gpu_res;
    cudaMalloc((void**)&d_input, numBytes);
    cudaMalloc((void**)&d_output, numBytes);
    cudaMalloc((void**)&d_final_res, sizeof(float));

    cudaMemcpy(d_input, h_input, numBytes, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    reduce_v2<<<num_blocks, BLOCKSIZE>>>(d_input, d_output);
    reduce_v2<<<1, num_blocks>>>(d_output, d_final_res);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    float gpu_time = 0;
    cudaEventElapsedTime(&gpu_time, start, stop);

    cudaMemcpy(&gpu_res, d_final_res, sizeof(float), cudaMemcpyDeviceToHost);

    std::cout<<"gpu result:"<<gpu_res<<std::endl;
    std::cout<<"gpu time:"<<gpu_time<<"ms"<<std::endl;

    if (abs(cpu_res - gpu_res) < 1e-5) {
        std::cout << "Result verified successfully!" << std::endl;
    } else {
        std::cout << "Result verification failed!" << std::endl;
    }

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_final_res);

    free(h_input);
    
}