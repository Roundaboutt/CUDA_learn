#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>

//---------------------------------------------------------------------------------------------------------------

// CPU中实现
void softmax_cpu(float* output, float* input, int N, int C){
    float maxval = -INFINITY;
    for (int i = 0;i < N; i++){
        const float* inp_row = input + C * i;

        for (int j = 0;j < C; j++){
            maxval = fmaxf(maxval, inp_row[j]);
        }
    }

    for (int i = 0;i < N; i++){
        const float* inp_row = input + C * i;
        float* out_row = output + C * i;
        
        float sum = 0.0f;
        for (int j = 0;j < C; j++){
            out_row[j] = expf(inp_row[j] - maxval);
            sum += out_row[j];
        }


        float norm = 1.0f / sum;
        for (int j = 0;j < C; j++){
            out_row[j] *= norm;
        }
    }


}

//---------------------------------------------------------------------------------------------------------------

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
//---------------------------------------------------------------------------------------------------------------

//利用规约操作和共享显存优化
__global__ void softmax_kernel2(float* output, float* input, int N, int C){

    // 声明共享显存
    extern __shared__ float shared[];
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    const float* input_row = input + bid * C;
    
    float maxval = -INFINITY;

    // 每个线程处理以下下标的数据:tid, tid + block_size, tid + 2 * block_size ...
    for (int i = tid; i < C;i += block_size){
        // 这里为什么不需要同步? 因为线程之间没有相互依赖
        maxval = fmaxf(maxval, input_row[i]);
    }
    // 每个线程负责的所有元素中的最大值 写入共享显存
    shared[tid] = maxval;
    __syncthreads();

    // 从局部最大值中找出全局最大值
    for (int stride = block_size / 2; stride >= 1; stride /= 2){

        // 等上一轮所有线程都完成之后再比较
        __syncthreads();
        if (tid < stride){
            shared[tid] = fmaxf(shared[tid], shared[tid + stride]);
        }
    }

    __syncthreads();
    float offset = shared[0];   // 最终的全局最大值

    for (int i = tid;i < C;i += block_size){
        output[bid * C + i] = expf(input_row[i] - offset);
    }

    const float* output_row = output + bid * C;
    float sumval = 0.0f;
    for (int i = tid;i < C;i += block_size){
        sumval += output_row[i];
    }

    // 索引为tid的线程所有负责元素的和
    shared[tid] = sumval;
    __syncthreads();

    // 规约计算全局和
    for (int stride = block_size / 2; stride >= 1; stride /= 2){
        __syncthreads();
        if (tid < stride){
            shared[tid] += shared[tid + stride];            
        }
    }

    __syncthreads();
    float sum = shared[0];
    for (int i = tid;i < C;i += block_size){
        output[bid * C + i] = output_row[i] / sum;
    }
}
//---------------------------------------------------------------------------------------------------------------

//利用warp洗牌指令优化
__device__ float warpReduceSum(float val){
    for (int offset = 16; offset >= 1; offset /= 2){
        val += __shfl_down_sync(0xffffffff, val, offset, 32);
    }
    return val;
}

__device__ float warpReduceMax(float val){
    for (int offset = 16; offset >= 1; offset /= 2){
        val = fmaxf(__shfl_down_sync(0xffffffff, val, offset, 32), val);
    }
    return val;
}


__global__ void softmax_kernel3(float* output, float* input, int N, int C){
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    
    const float* input_row = input + bid * C;
    float maxval = -INFINITY;
    for (int i = tid; i < C; i += block_size){
        maxval = fmaxf(maxval, input_row[i]);
    }

    maxval = warpReduceMax(maxval);
    float offset = __shfl_sync(0xffffffff, maxval, 0);

    float* output_row = output + bid * C;
    for (int i = tid;i < C; i += block_size){
        output_row[i] = expf(input_row[i] - offset);
    }

    float sumval = 0.0f;
    for (int i = tid; i < C; i += block_size){
        sumval += output_row[i];
    }
    sumval = warpReduceSum(sumval);

    float sum = __shfl_sync(0xffffffff, sumval, 0);
    for (int i = tid;i < C; i += block_size){
        output_row[i] /= sum;
    }

}





//---------------------------------------------------------------------------------------------------------------
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

    softmax_cpu(output, input, N, C);


    for (int i = 0;i < 16; i++){
        printf("%.10f\n", output[i]);
    }
    printf("--------------------------------------------------\n");
    float* d_input,* d_output;
    cudaMalloc((void**)&d_input, elemCount*sizeof(float));
    cudaMalloc((void**)&d_output, elemCount*sizeof(float));
    cudaMemcpy(d_input, input, elemCount*sizeof(float), cudaMemcpyHostToDevice);


    int blockSize = 32; 
    int numBlocks = N;  // N个线程块,每个线程块负责一个向量

    softmax_kernel3<<<numBlocks, blockSize>>>(d_output, d_input, N, C);

    cudaMemcpy(output, d_output, elemCount*sizeof(float), cudaMemcpyDeviceToHost);


    for (int i = 0;i < 16; i++){
        printf("%.10f\n", output[i]);
    }

}