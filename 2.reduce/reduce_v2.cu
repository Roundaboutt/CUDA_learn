#include<iostream>

#define N 1024*1024*16   // 可以很大
#define BLOCKSIZE 1024   // 每个 block 最大线程数
#define FULLMASK 0xFFFFFFFF

__device__ float BlockReduce(float val){
    const int tid = threadIdx.x;
    const int warpID = tid / warpSize;
    const int laneID = tid % warpSize;

#pragma unroll      
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
    {
        val += __shfl_down_sync(FULLMASK, val, offset);
    }

    __shared__ float shared[32];
    if (laneID == 0)
    {
        shared[warpID] = val;
    }

    __syncthreads();

    if (warpID == 0)
    {
        val = shared[laneID];
#pragma unroll          
        for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        {
            val += __shfl_down_sync(FULLMASK, val, offset);
        }
    }

    return val;
}

__global__ void reduce_v2(float* input, float* output, int n){
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const int global_id = tid + blockDim.x * bid;

    const int packSize = 4;
    const int pack_num = n / packSize;
    const int pack_off = pack_num * packSize;

    float sum = 0.f;
    float4* input_pack = reinterpret_cast<float4*> (input);

#pragma unroll   
    for (int i = global_id; i < pack_num; i += blockDim.x * gridDim.x)
    {
        float4 in = *(input_pack + i);
        sum += in.x;
        sum += in.y;
        sum += in.z;
        sum += in.w;
    }

#pragma unroll  
    for (int i = global_id + pack_off; i < n; i += blockDim.x * gridDim.x)
    {
        sum += input[i];
    }

    sum = BlockReduce(sum);
    if (tid == 0)
        output[bid] = sum;
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
        h_input[i] = 2.0f;
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
        reduce_v2<<<blocks, threads>>>(d_in, d_out, n);
        cudaDeviceSynchronize();

        n = blocks;
        blocks = (n + threads * 2 - 1) / (threads * 2);

        // 交换 in/out
        float* tmp = d_in;
        d_in = d_out;
        d_out = tmp;
    }

    // 最后一层
    reduce_v2<<<1, threads>>>(d_in, d_out, n);

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
