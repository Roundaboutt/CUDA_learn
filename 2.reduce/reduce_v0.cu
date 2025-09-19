#include <iostream>
#include "cuda_runtime.h"
#include <chrono>

#define N 1024*1024*10   // 可以很大
#define BLOCKSIZE 1024   // 每个 block 最大线程数

__global__ void reduce_v0(float* input, float* output, int n){
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int id = tid + bid * blockDim.x;

    extern __shared__ float shared[];
    
    // 拷贝数据到共享内存
    if (id < n){
        shared[tid] = input[id];
    }
    else{
        shared[tid] = 0.0f;
    }

    __syncthreads();
    
    // 归约
    for (int s = 1; s < blockDim.x; s <<= 1){
        if (tid % (s * 2) == 0 && tid + s < blockDim.x){
            shared[tid] += shared[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0){
        output[bid] = shared[0];
    }
}

float reduce_cpu(float* input){
    float sum = 0.0f;
    for (int i = 0;i < N; i++){
        sum += input[i];
    }
    return sum;
}

int main(){
    const int elemCount = N;
    const int numBytes = N * sizeof(float);

    float* h_input = (float*)malloc(numBytes);

    for (int i = 0;i < elemCount; i++){
        h_input[i] = 1.0f;
    }

    // CPU baseline
    auto cpu_start = std::chrono::high_resolution_clock::now();
    float cpu_res = reduce_cpu(h_input);
    auto cpu_end = std::chrono::high_resolution_clock::now();    
    std::chrono::duration<double, std::milli> cpu_time = cpu_end - cpu_start;

    std::cout<<"cpu result:"<<cpu_res<<std::endl;
    std::cout<<"cpu time:"<<cpu_time.count()<<"ms"<<std::endl;

    // GPU
    float *d_input,* d_output;
    float gpu_res;
    cudaMalloc((void**)&d_input, numBytes);
    cudaMalloc((void**)&d_output, numBytes); // 开大点，逐层写结果

    cudaMemcpy(d_input, h_input, numBytes, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    int threads = BLOCKSIZE;
    int n = N;
    int blocks = (n + threads - 1) / threads;
    float* d_in = d_input;
    float* d_out = d_output;

    // 逐层归约
    while (blocks > 1){
        reduce_v0<<<blocks, threads, threads * sizeof(float)>>>(d_in, d_out, n);
        n = blocks;
        blocks = (n + threads - 1) / threads;

        // 交换 in/out
        float* tmp = d_in;
        d_in = d_out;
        d_out = tmp;
    }

    // 最后一层
    reduce_v0<<<1, threads, threads * sizeof(float)>>>(d_in, d_out, n);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpu_time = 0;
    cudaEventElapsedTime(&gpu_time, start, stop);

    cudaMemcpy(&gpu_res, d_out, sizeof(float), cudaMemcpyDeviceToHost);

    std::cout<<"gpu result:"<<gpu_res<<std::endl;
    std::cout<<"gpu time:"<<gpu_time<<"ms"<<std::endl;

    if (abs(cpu_res - gpu_res) < 1e-5) {
        std::cout << "Result verified successfully!" << std::endl;
    } else {
        std::cout << "Result verification failed!" << std::endl;
    }

    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
}
