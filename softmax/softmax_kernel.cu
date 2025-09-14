#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>

// N个向量,每个向量中有C个元素
__global__ void softmax_kernel1(float* output, float* input, int N, int C){

    // N个线程块,每个线程块负责一个向量的计算
    // 每个线程块中只有一个线程计算
    // id:全局线程索引
    int id = blockIdx.x * blockDim.x + threadIdx.x; // id:[0, N-1)  blockIdx:[0, N-1)  threadIdx:0
    if (id < N){
        const float* input_row = input + id*C;
        float* output_row = output + id*C;

        float maxval = -INFINITY;
        for (int i = 0;i < C;i++){
            if (input_row[i] > maxval){
                maxval = input_row[i];
            }
        }

        float sum = 0.f;
        for (int j = 0;j < C;j++){
            output_row[j] = expf(input_row[j] - maxval);
            sum += output_row[j];
        }

        for (int j = 0;j < C;j++){
            output_row[j] /= sum;
        }
    }

}


int main(){
    int N = 32;
    int C = 4096;
    size_t elemCount = N * C;

    float* input = (float*)malloc(sizeof(float)*elemCount);
    float* output = (float*)malloc(sizeof(float)*elemCount);

    for (int n = 0;n < N;n++){
        for(int c = 0;c < C;c++){
            input[n*C + c] = float(0);
        }
    }

    float* d_input,* d_output;
    cudaMalloc((void**)&d_input, elemCount*sizeof(float));
    cudaMalloc((void**)&d_output, elemCount*sizeof(float));
    cudaMemcpy(d_input, input, elemCount*sizeof(float), cudaMemcpyHostToDevice);


    int blockSize = 1;  // 每个线程块中只有一个线程参与运算
    int numBlocks = N;  // N个线程块,每个线程块负责一个向量

    softmax_kernel1<<<numBlocks, blockSize>>>(d_output, d_input, N, C);

    cudaMemcpy(output, d_output, elemCount*sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0;i < elemCount; i++){
        printf("%.10f\n", output[i]);
    }
}