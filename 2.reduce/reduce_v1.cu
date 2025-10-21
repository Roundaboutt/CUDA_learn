#include <iostream>
#include "cuda_runtime.h"

#define N 1024*1024*16   // 可以很大
#define BLOCKSIZE 1024   // 每个 block 最大线程数
#define FULLMASK 0xFFFFFFFF


__device__ float BlockReduce(float val){
    const int tid = threadIdx.x;
    const int warpID = tid / warpSize;
    const int laneID = tid % warpSize;

    // warp内的归约
    for (int offset = warpSize / 2; offset > 0; offset >>= 1){
        val += __shfl_down_sync(FULLMASK, val, offset);
    }

    // 共享显存用来存储每个warp的归约结果
    __shared__ float warpShared[32];
    if (laneID == 0)
        warpShared[warpID] = val;
    
    __syncthreads();


    // 用0号warp来归约整个共享显存(即warp间的归约)
    if (warpID == 0){
        val = warpShared[laneID];
        for (int offset = warpSize / 2; offset > 0; offset >>= 1){
            val += __shfl_down_sync(FULLMASK, val, offset);
        }
    }

    return val;
}

__global__ void reduce_v1(float* input, float* output, int n){
    const int global_id = threadIdx.x + blockDim.x * blockIdx.x;
    
    float sum = 0.f;
    for (int i = global_id; i < n; i += blockDim.x * gridDim.x){
        sum += input[i];
    }

    sum = BlockReduce(sum);

    if (threadIdx.x == 0)
        output[blockIdx.x] = sum;
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
        reduce_v1<<<blocks, threads>>>(d_in, d_out, n);
        cudaDeviceSynchronize();

        n = blocks;
        blocks = (n + threads * 2 - 1) / (threads * 2);

        // 交换 in/out
        float* tmp = d_in;
        d_in = d_out;
        d_out = tmp;
    }

    // 最后一层
    reduce_v1<<<1, threads>>>(d_in, d_out, n);

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
