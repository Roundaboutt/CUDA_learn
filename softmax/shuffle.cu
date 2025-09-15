#include <stdio.h>
#include "cuda_runtime.h"
#include <iostream>


__global__ void test_shuffle_broadcast(int* d_output, const int* d_input){
    int val = d_input[threadIdx.x];
    val = __shfl_sync(0xFFFFFFFF, val, 2, 32);
    d_output[threadIdx.x] = val;
}


__global__ void test_shuffle_down(int* d_output, const int* d_input){
    int val = d_input[threadIdx.x];

    // 0xffffffff:32位掩码,表示warp中32个所有线程都参加
    // val:每个线程手里的变量
    // delta(2):向下偏移多少个lane
    // width(32):warp的宽度,默认32
    // 返回值:当前线程从 比自己高 delta 个 lane 的线程那里,拿到那个线程的 var 值。
    val = __shfl_down_sync(0xffffffff, val, 2, 32);
    d_output[threadIdx.x] = val;
}


__global__ void test_shuffle_up(int* d_output, const int* d_input){
    int val = d_input[threadIdx.x];
    // 当前线程从 比自己低 delta 个 lane 的线程那里,拿到那个线程的 var 值。
    val = __shfl_up_sync(0xffffffff, val, 2, 32);
    d_output[threadIdx.x] = val;
}


int main() {

    const int numThreads = 64;  //如果这里超过了32,每个warp内会独立通信

    int* host_input = (int*)malloc(sizeof(int) * numThreads);
    int* host_output = (int*)malloc(sizeof(int) * numThreads);

    // 初始化
    for (int i = 0;i < numThreads; i++){
        host_input[i] = i;
    }

    int* device_input,* device_output;
    cudaMalloc((void**)&device_input, sizeof(int) * numThreads);
    cudaMalloc((void**)&device_output, sizeof(int) * numThreads);

    cudaMemcpy(device_input, host_input, sizeof(int) * numThreads, cudaMemcpyHostToDevice);

    test_shuffle_broadcast<<<1, numThreads>>>(device_output, device_input);
    cudaDeviceSynchronize();

    cudaMemcpy(host_output, device_output, sizeof(int) * numThreads, cudaMemcpyDeviceToHost);
    for (int i = 0; i < numThreads; i++){
        std::cout<<"host_output["<< i <<"]="<<host_output[i]<<std::endl;
    }

}