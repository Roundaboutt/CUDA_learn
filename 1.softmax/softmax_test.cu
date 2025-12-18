#include<iostream>
#include"cuda_runtime.h"
#include<chrono>
#include<cstdlib>


void cpu_kernel(float* input, float* output, int row, int col){
    for (int i = 0; i < row; i++){
        float maxval = -INFINITY;
        float* input_row = input + col * i;
        float* output_row = output + col * i;

        for (int j = 0; j < col; j++){
            maxval = fmaxf(maxval, input_row[j]);
        }

        float sum = 0.0f;
        for (int j = 0; j < col; j++){
            output_row[j] = expf(input_row[j] - maxval);
            sum += output_row[j];
        }

        for (int j = 0; j < col; j++){
            output_row[j] /= sum;
        }
    }
}

#define FULLMASK 0xFFFFFFFF

__device__ float MaxReduce(float val)
{
    const int tid = threadIdx.x;
    const int warpID = tid / warpSize;
    const int laneID = tid % warpSize;

#pragma unroll    
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
    {
        val = fmaxf(__shfl_down_sync(FULLMASK, val, offset), val);
    }

    __shared__ float warp_shared[32];

    if (laneID == 0)
    {
        warp_shared[warpID] = val;
    }

    __syncthreads();

    if (warpID == 0)
    {
        val = warp_shared[laneID];
#pragma unroll        
        for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        {
            val = fmaxf(val, __shfl_down_sync(FULLMASK, val, offset));
        }
    }

    return val;
}

__device__ float SumReduce(float val)
{
    const int tid = threadIdx.x;
    const int warpID = tid / warpSize;
    const int laneID = tid % warpSize;

#pragma unroll    
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
    {
        val += __shfl_down_sync(FULLMASK, val, offset);
    }

    __shared__ float warp_shared[32];
    if (laneID == 0)
    {
        warp_shared[warpID] = val;
    }
    __syncthreads();

    if (warpID == 0)
    {
        val = warp_shared[laneID];
#pragma unroll        
        for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        {
            val += __shfl_down_sync(FULLMASK, val, offset);
        }
    }

    return val;
}


__global__ void softmax_kernel(float* input, float* output, int row, int col)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int global_id = tid + blockDim.x * bid;

    float* input_start = input + col * bid;
    float* output_start = output + col * bid;

    float max = -INFINITY;

#pragma unroll    
    for (int i = tid; i < col; i += blockDim.x)
    {
        max = fmax(max, input_start[i]);
    }

    max = MaxReduce(max);
    __shared__ float shared_max;
    if (tid == 0)
        shared_max = max;
    __syncthreads();
    max = shared_max;

    float sum = 0.f;

#pragma unroll    
    for (int i = tid; i < col; i += blockDim.x)
    {
        sum += expf(input_start[i] - max);
    }
    
    sum = SumReduce(sum);
    __shared__ float shared_sum;
    if (tid == 0)
        shared_sum = sum;
    __syncthreads();
    sum = shared_sum;

#pragma unroll    
    for (int i = tid; i < col; i += blockDim.x)
    {
        output_start[i] = expf(input_start[i] - max) / sum;
    }
}


bool compare(float* h_output, float* d_output, const int elemCount){
    for (int i = 0; i < elemCount; i++){
        if (abs(h_output[i] - d_output[i]) > 1e-3){
            std::cout<<"cpu res:"<<h_output[i]<<std::endl;
            std::cout<<"gpu res:"<<d_output[i]<<std::endl;
            return false;
        }
    }
    return true;
}


int main(){
    constexpr int row = 1024 * 20;
    constexpr int col = 16384;
    constexpr size_t elemCount = row * col;
    constexpr size_t numBytes = elemCount * sizeof(float);

    float* h_input = (float*)malloc(numBytes);
    float* h_output = (float*)malloc(numBytes);

    for (int i = 0; i < row; i++){
        for (int j = 0; j < col; j++){
            h_input[i * col + j] = rand() % 10;
        }
    }

    auto cpu_start = std::chrono::high_resolution_clock::now();
    cpu_kernel(h_input, h_output, row, col);
    auto cpu_end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> cpu_time = cpu_end - cpu_start;
    std::cout<<"cpu time:"<<cpu_time.count()<<"ms"<<std::endl;

    float* d_input,* d_output;
    cudaMalloc((void**)&d_input, numBytes);
    cudaMalloc((void**)&d_output, numBytes);
    cudaMemcpy(d_input, h_input, numBytes, cudaMemcpyHostToDevice);

    constexpr int blockSize = 1024;
    constexpr int numBlocks = row;

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    cudaEventRecord(start);
    softmax_kernel<<<numBlocks, blockSize>>>(d_input, d_output, row, col);
    cudaEventRecord(end);

    cudaEventSynchronize(end);

    float gpu_time = 0;
    cudaEventElapsedTime(&gpu_time, start, end);

    cudaEventDestroy(start);
    cudaEventDestroy(end);
    std::cout<<"gpu time:"<<gpu_time<<"ms"<<std::endl;

    float* gpu_output = (float*)malloc(numBytes);
    cudaMemcpy(gpu_output, d_output, numBytes, cudaMemcpyDeviceToHost);

    if (compare(h_output, gpu_output, elemCount)){
        std::cout<<"successful!"<<std::endl;
    }
    else{
        std::cout<<"not successful!"<<std::endl;
    }

    std::cout<<"speed up:"<<(cpu_time.count() / gpu_time)<<"x"<<std::endl;
}