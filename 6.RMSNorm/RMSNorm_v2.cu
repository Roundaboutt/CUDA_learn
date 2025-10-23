#include<iostream>
#include<cuda_runtime.h>
#include<random>

#define FULLMASK 0xFFFFFFFF

__device__ float BlockReduce(float val){
    const int tid = threadIdx.x;
    const int warpID = tid / warpSize;
    const int laneID = tid % warpSize;
    
    #pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset >>= 1){
        val += __shfl_down_sync(FULLMASK, val, offset);
    }

    __shared__ float warpShared[32];
    if (laneID == 0)
        warpShared[warpID] = val;
    
    __syncthreads();

    if (warpID == 0){
        val = warpShared[laneID];
        #pragma unroll
        for (int offset = warpSize / 2; offset > 0; offset >>= 1){
            val += __shfl_down_sync(FULLMASK, val, offset);
        }

    }

    return val;
}

__global__ void rmsnorm_v2(float* input, float* output, int batch, int size, float* weight, float eps){
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;

    if  (bid >= batch) return;
    
    float* input_start = input + size * bid;
    float* output_start = output + size * bid;

    constexpr int pack_size = 4;
    const int pack_num = size / pack_size;
    // 剩余部分:(pack_off, size)
    const int pack_off = pack_num * pack_size;

    float sum = 0.f;
    float4* in_pack = reinterpret_cast<float4*>(input_start);
    for (int i = tid; i < pack_num; i += blockDim.x){
        float4 in_float4 = *(in_pack + i);
        sum += in_float4.x * in_float4.x;
        sum += in_float4.y * in_float4.y;
        sum += in_float4.z * in_float4.z;
        sum += in_float4.w * in_float4.w;
    }

    for (int i = pack_off + tid; i < size; i += blockDim.x){
        sum += input_start[i] * input_start[i];
    }

    sum = BlockReduce(sum);
    __shared__ float shared_val;
    if (tid == 0) shared_val = sum;

    __syncthreads();

    sum = shared_val;

    const float scale = rsqrtf(sum / static_cast<float>(size) + eps);
    float4* out_pack = reinterpret_cast<float4*>(output_start);
    float4* wei_pack = reinterpret_cast<float4*>(weight);
    for (int i = tid; i < pack_num; i += blockDim.x){
        float4 wei_float4 = *(wei_pack + i);
        float4 in_float4 = *(in_pack + i);

        *(out_pack + i)= make_float4(
            scale * wei_float4.x * in_float4.x,
            scale * wei_float4.y * in_float4.y,
            scale * wei_float4.z * in_float4.z,
            scale * wei_float4.w * in_float4.w
        );
    }

    for (int i = pack_off + tid; i < size; i += blockDim.x){
        output_start[i] = scale * weight[i] * input_start[i];
    }
}

bool isMatch(float* output_cpu, float* output_gpu, int elemCount){
    for (int i = 0; i < elemCount; i++){
        if (abs(output_cpu[i] - output_gpu[i]) > 1e-5){
            return false;
        }
    }
    return true;
}


void rmsnorm_cpu(float* input, float* output, int batch, int size, float* weight, float eps){
    for (int i = 0; i < batch; i++){
        float* input_start = input + size * i;
        float* output_start = output + size * i;

        float sum = 0.f;
        for (int j = 0; j < size; j++){
            sum += input_start[j] * input_start[j];
        }

        float rms = 1.f / std::sqrt(sum / static_cast<float>(size) + eps);

        for (int j = 0; j < size; j++){
            output_start[j] = input_start[j] * weight[j] * rms;
        }
    }
}


int main(){
    const int batch = 16;
    const int size = 1024;
    const float eps = 1e-6;
    const int elemCount = batch * size;
    const int numBytes = elemCount * sizeof(float);

    float* h_input,* h_output_cpu,* h_output_gpu,* h_weight;
    h_input = (float*)malloc(numBytes);
    h_output_cpu = (float*)malloc(numBytes);
    h_output_gpu = (float*)malloc(numBytes);
    h_weight = (float*)malloc(sizeof(float) * size);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dis(0.0f, 1.0f);

    for (int i = 0; i < elemCount; i++){
        h_input[i] = dis(gen);
    }

    for (int i = 0; i < size; i++){
        h_weight[i] = dis(gen);
    }

    rmsnorm_cpu(h_input, h_output_cpu, batch, size, h_weight, eps);

    // ------------------------------

    float* d_input,* d_output,* d_weight;
    cudaMalloc((void**)&d_input, numBytes);
    cudaMalloc((void**)&d_output, numBytes);
    cudaMalloc((void**)&d_weight, sizeof(float) * size);

    cudaMemcpy(d_input, h_input, numBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, h_weight, sizeof(float) * size, cudaMemcpyHostToDevice);

    const int blocksize = 1024;
    const int gridsize = batch;
    dim3 grid(gridsize);
    dim3 block(blocksize);

    rmsnorm_v2<<<grid, block>>>(d_input, d_output, batch, size, d_weight, eps);

    cudaMemcpy(h_output_gpu, d_output, numBytes, cudaMemcpyDeviceToHost);

    if (isMatch(h_output_cpu, h_output_gpu, elemCount)){
        std::cout << "Match!" << std::endl;
    }
    else{
        std::cout << "Not Match!" << std::endl;
    }

}