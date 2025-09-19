#include <cuda_runtime.h>

#include <chrono>
#include <iostream>
#include <numeric>
#include <vector>

#define N 1024 * 1024
#define BLOCKSIZE 1024
#define FULLMASK 0xFFFFFFFF


__inline__ __device__ float blockReduce(float val){
    int tid = threadIdx.x;

    const int warpSize = 32;
    const int warpID = tid / warpSize;
    const int laneID = tid % warpSize;

#pragma unroll
    // warp 内的归约
    for (int offset = warpSize / 2; offset > 0; offset >>= 1){
        val += __shfl_down_sync(FULLMASK, val, offset);
    }

    __shared__ float warpShared[32];
    if (laneID == 0) warpShared[warpID] = val;

    __syncthreads();

    if (warpID == 0){
        val = (laneID < blockDim.x / warpSize) ? warpShared[laneID]:0.0f;
        
#pragma unroll
        for (int offset = warpSize / 2; offset > 0; offset >>= 1){
            val += __shfl_down_sync(FULLMASK, val, offset);
        }
    }
    
    return val;
}


__global__ void reduce_v4(float* input, float* output, int n){
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int id = tid + bid * blockDim.x;
    float sum = 0.f;

    for (int i = id; i < n; i += gridDim.x * blockDim.x){
        sum += input[i];
    }

    sum = blockReduce(sum);

    if (tid == 0){
        output[bid] = sum;
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
    const int num_blocks = ((N + BLOCKSIZE - 1) / BLOCKSIZE);

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
    reduce_v4<<<num_blocks, BLOCKSIZE>>>(d_input, d_output, N);
    reduce_v4<<<1, num_blocks>>>(d_output, d_final_res, num_blocks);
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