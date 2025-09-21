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


__global__ void softmax_kernel(float* input, float* output, int row, int col){
    extern __shared__ float shared[];
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    float* input_row = input + col * bid;

    float maxval = -INFINITY;
    for (int i = tid; i < col; i += blockDim.x){
        maxval = fmaxf(maxval, input[i]);
    }
    
    // 把每个线程负责的所有元素中的最大值写入共享显存
    shared[tid] = maxval;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1){
        
        __syncthreads();
        if (tid < s){
            shared[tid] = fmaxf(shared[tid], shared[tid + s]);            
        }

    }

    __syncthreads();

    float offset = shared[0];
    float* output_row = output + col * bid;
    float sumval = 0.0f;
    for (int i = tid; i < col; i += blockDim.x){
        output_row[i] = expf(input_row[i] - offset);
        sumval += output_row[i];
    }
    shared[tid] = sumval;

    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1){
        __syncthreads();
        if (tid < s){
            shared[tid] += shared[tid + s];
        }
    }

    float sum = shared[0];

    for (int i = tid; i < col; i += blockDim.x){
        output_row[i] /= sum;
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
    int row = 1024 * 20;
    int col = 16384;
    size_t elemCount = row * col;
    size_t numBytes = elemCount * sizeof(float);

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

    int blockSize = 1024;
    int numBlocks = row;

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    cudaEventRecord(start);
    softmax_kernel<<<numBlocks, blockSize, blockSize * sizeof(float)>>>(d_input, d_output, row, col);
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