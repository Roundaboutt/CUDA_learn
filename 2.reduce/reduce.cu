#include <stdio.h>
#include "cuda_runtime.h"
#include <iostream>
#include <vector>
#include <chrono>


const int BLOCK_SIZE = 1024;
const int N = 1024 * 1024;

#define FULL_MASK 0xFFFFFFFF

// -------------------------------------------------------------------------------------
__global__ void reduce_v0(float* input, float* output){
    __shared__ float shared[BLOCK_SIZE];

    unsigned int tid = threadIdx.x;
    unsigned int bid = blockIdx.x;
    unsigned int id = tid + blockDim.x * bid;

    

    if (id < N){
        shared[tid] = input[id];    // 把当前线程的元素复制到共享内存        
    }
    else{
        shared[tid] = 0.f;
    }

    __syncthreads();
    for (unsigned int s = 1; s < blockDim.x; s *= 2){
        __syncthreads();
        if (tid % (2 * s) == 0){
            shared[tid] += shared[tid + s]; 
        }
        
    }
    
    __syncthreads();
    // 该线程块的局部规约结果
    if (tid == 0) output[bid] = shared[0]; 
}

// -------------------------------------------------------------------------------------
// 现在全局显存做归约,解决线程闲置的问题
__global__ void reduce_v1(float* input, float* output){
    __shared__ float shared[BLOCK_SIZE];

    unsigned int tid = threadIdx.x;
    unsigned int bid = blockIdx.x;
    unsigned int id = tid + 2 * blockDim.x * bid;

    

    if (id < N){
        shared[tid] = input[id] + input[id + blockDim.x];    // 先在全局显存做一次归约       
    }
    else{
        shared[tid] = 0.f;
    }

    __syncthreads();
    for (unsigned int s = 1; s < blockDim.x; s *= 2){
        __syncthreads();
        if (tid % (2 * s) == 0){
            shared[tid] += shared[tid + s]; 
        }
        
    }
    
    __syncthreads();
    // 该线程块的局部规约结果
    if (tid == 0) output[bid] = shared[0]; 
}


// -------------------------------------------------------------------------------------
// 解决线程束分化的问题

__global__  void reduce_v2(const float* input, float* output){
    __shared__ float shared[BLOCK_SIZE];
    unsigned int bid = blockIdx.x;
    unsigned int tid = threadIdx.x;
    unsigned int id = tid + bid * blockDim.x * 2;
    
    shared[tid] = input[id] + input[id + blockDim.x];

    __syncthreads();

    for (int s = blockDim.x / 2; s >= 1; s >>= 1){
        __syncthreads();

        // 避免了线程束分化
        // 但如果 s <= 32 ,还是会出现线程束分化
        if (tid < s){
            shared[tid] += shared[tid + s];
        }
    }

    if (tid == 0) output[bid] = shared[0];
}
// -------------------------------------------------------------------------------------
__device__ void warpReduce(float* cache, unsigned int tid){
    float val = cache[tid] + cache[tid + 32];
    for (int offset = 16; offset >= 1; offset /= 2){
        val += __shfl_down_sync(FULL_MASK, val, offset);
    }

    cache[tid] = val;
}

__global__ void reduce_v3(const float* input, float* output){
    __shared__ float shared[BLOCK_SIZE];
    unsigned int bid = blockIdx.x;
    unsigned int tid = threadIdx.x;
    unsigned int id = tid + blockDim.x * bid * 2;

    shared[tid] = input[id] + input[id + blockDim.x];

    __syncthreads();

    for (int s = blockDim.x / 2; s > 32; s >>= 1){
        __syncthreads();
        if (tid < s){
            shared[tid] += shared[tid + s];
        }
    }

    if (tid < 32){
        warpReduce(shared, tid);
    }
    if (tid == 0){
        output[bid] = shared[0];
    }
}

// -------------------------------------------------------------------------------------
float reduce_cpu(const std::vector<float> &data){
    float sum = 0.f;
    for (float val : data){
        sum += val;
    }
    return sum;
}
// -------------------------------------------------------------------------------------

int main(){

    int num_blocks = ((N + BLOCK_SIZE - 1) / BLOCK_SIZE) / 2;

    std::vector<float> h_data(N);

    for (int i = 0; i < N; i++){
        h_data[i] = 1.f;
    }

    auto cpu_start = std::chrono::high_resolution_clock::now();
    float cpu_res = reduce_cpu(h_data);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_duration = cpu_end - cpu_start;

    std::cout<<"cpu result:"<<cpu_res<<std::endl;
    std::cout<<"cpu time:"<<cpu_duration.count()<<"ms"<<std::endl;



    float* d_input,* d_output;
    float* d_final_output;
    float gpu_res;

    cudaMalloc((void**)&d_input, N * sizeof(float));    // 输入的数组
    cudaMalloc((void**)&d_output, num_blocks * sizeof(float));  // 第一步归约的结果,计算每个block的总和
    cudaMalloc((void**)&d_final_output, 1 * sizeof(float));     //最后归约的结果,所有block的和
    cudaMemcpy(d_input, h_data.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    reduce_v3<<<num_blocks, BLOCK_SIZE>>>(d_input, d_output);
    reduce_v3<<<1, num_blocks>>>(d_output, d_final_output);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    

    float gpu_time = 0;
    cudaEventElapsedTime(&gpu_time, start, stop);
    cudaMemcpy(&gpu_res, d_final_output, sizeof(float), cudaMemcpyDeviceToHost);  

    std::cout<<"gpu result:"<<gpu_res<<std::endl;
    std::cout<<"gpu time:"<<gpu_time<<"ms"<<std::endl;
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_output);
    cudaFree(d_final_output);
    cudaFree(d_input);
}