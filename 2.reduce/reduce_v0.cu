#include <iostream>
#include "cuda_runtime.h"
#include <chrono>

#define N 1024*1024*10   // 可以很大
#define BLOCKSIZE 1024   // 每个 block 最大线程数

__device__ void warpReduce(volatile float* cache, const int tid){
    cache[tid] += cache[tid + 32];
    cache[tid] += cache[tid + 16];
    cache[tid] += cache[tid + 8];
    cache[tid] += cache[tid + 4];
    cache[tid] += cache[tid + 2];
    cache[tid] += cache[tid + 1];
}

__global__ void reduce_v0(float* input, float* output, int n){
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const int global_id = tid + bid * blockDim.x * 2;

    __shared__ float shared[BLOCKSIZE];

    if (global_id + blockDim.x < n){
        shared[tid] = input[global_id] + input[global_id + blockDim.x];
    }
    else if (global_id < n){
        shared[tid] = input[global_id];
    }
    else{
        shared[tid] = 0;
    }

    __syncthreads();

    #pragma unroll
    for (int s = blockDim.x / 2; s > 32; s >>= 1){
        if (tid < s){
            shared[tid] += shared[tid + s];
        }

        __syncthreads();

    }

    if (tid < 32){
        warpReduce(shared, tid);
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

    float cpu_res = reduce_cpu(h_input);

    // GPU
    float *d_input,* d_output;
    float gpu_res;
    cudaMalloc((void**)&d_input, numBytes);
    cudaMalloc((void**)&d_output, numBytes);

    cudaMemcpy(d_input, h_input, numBytes, cudaMemcpyHostToDevice);

    int threads = BLOCKSIZE;
    int n = N;
    int blocks = (n + threads * 2 - 1) / (threads * 2);
    float* d_in = d_input;
    float* d_out = d_output;

    // 逐层归约
    while (blocks > 1){
        reduce_v0<<<blocks, threads>>>(d_in, d_out, n);
        cudaDeviceSynchronize();

        n = blocks;
        blocks = (n + threads * 2 - 1) / (threads * 2);

        // 交换 in/out
        float* tmp = d_in;
        d_in = d_out;
        d_out = tmp;
    }

    // 最后一层
    reduce_v0<<<1, threads>>>(d_in, d_out, n);

    cudaMemcpy(&gpu_res, d_out, sizeof(float), cudaMemcpyDeviceToHost);

    if (abs(cpu_res - gpu_res) < 1e-5) {
        std::cout << "Result verified successfully!" << std::endl;
    } else {
        std::cout << "Result verification failed!" << std::endl;
    }

    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
}
