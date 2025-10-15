#include<iostream>
#include"cuda_runtime.h"


float* transpose_cpu(float* output, int nx, int ny){

    float input[nx * ny];
    for (int i = 0; i < nx * ny; i++){
        input[i] = float(i);
    }

    for (int i = 0; i < ny; i++){
        for (int j = 0; j < nx; j++){
            // output[j][i] = input[i][j];
            output[j * ny + i] = input[i * nx + j];
        }
    }
    return output;
}


#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])
template <const int THREAD_SIZE_Y, const int THREAD_SIZE_X>
__global__ void transpose_v2(float* output, float* input, int M, int N){
    float src[4][4];
    float dst[4][4];
    float* input_start = blockIdx.y * THREAD_SIZE_Y * N + blockIdx.x * THREAD_SIZE_X + input;
    for (int i = 0; i < 4; i++){
        FETCH_FLOAT4(src[i]) = FETCH_FLOAT4(input_start[(threadIdx.y * 4 + i) * N + threadIdx.x * 4]);
    }

    FETCH_FLOAT4(dst[0]) = make_float4(src[0][0], src[1][0], src[2][0], src[3][0]);
    FETCH_FLOAT4(dst[1]) = make_float4(src[0][1], src[1][1], src[2][1], src[3][1]);
    FETCH_FLOAT4(dst[2]) = make_float4(src[0][2], src[1][2], src[2][2], src[3][2]);
    FETCH_FLOAT4(dst[3]) = make_float4(src[0][3], src[1][3], src[2][3], src[3][3]);

    float* output_start = blockIdx.x * THREAD_SIZE_X * M + blockIdx.y * THREAD_SIZE_Y + output;

    for (int i = 0; i < 4; i++){
        FETCH_FLOAT4(output_start[(threadIdx.x * 4 + i) * M + threadIdx.y * 4]) = FETCH_FLOAT4(dst[i]);
    }
}

template <const int BLKDIM_x, const int BLKDIM_y> 
void call_v2(float* d_output, float* d_input, int M, int N){
    dim3 blockSize(BLKDIM_x, BLKDIM_y);  // 每个线程块有 32x8 = 256 个线程
    constexpr int THREAD_SIZE_X = BLKDIM_x * 4;  // 每个 block 处理 128 列
    constexpr int THREAD_SIZE_Y = BLKDIM_y * 4;  // 每个 block 处理 32 行

    dim3 gridSize(
        (N + THREAD_SIZE_X - 1) / THREAD_SIZE_X,
        (M + THREAD_SIZE_Y - 1) / THREAD_SIZE_Y
    );

    transpose_v2<THREAD_SIZE_Y, THREAD_SIZE_X><<<gridSize, blockSize>>>(d_output, d_input, M, N);
}

template <const int BLKDIM_x, const int BLKDIM_y>
float* v2_time(int M, int N, int warmup, int repeat_time){

    int elemCount = M * N;
    int numBytes = elemCount * sizeof(float);

    float* h_input,* h_output;
    h_input = (float*)malloc(numBytes);
    h_output = (float*)malloc(numBytes);

    for (int i = 0; i < elemCount; i++){
        h_input[i] = float(i);
    }


    float* d_input,* d_output;
    cudaMalloc((void**)&d_input, numBytes);
    cudaMalloc((void**)&d_output, numBytes);

    cudaMemcpy(d_input, h_input, numBytes, cudaMemcpyHostToDevice);

    for (int i = 0; i < warmup; i++){
        call_v2<BLKDIM_x, BLKDIM_y>(d_output, d_input, M, N);        
    }


    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    cudaEventRecord(start);
    for (int i = 0; i < repeat_time; i++){
        call_v2<BLKDIM_x, BLKDIM_y>(d_output, d_input, M, N);
    }
    cudaEventRecord(end);
    cudaEventSynchronize(end);

    float v2_time = 0;
    cudaEventElapsedTime(&v2_time, start, end);
    std::cout << "blockdim:(" << BLKDIM_x << "," << BLKDIM_y << ")" << std::endl;
    std::cout << "v2 time:" << v2_time << "ms" << std::endl;
    std::cout << "---------------------" << std::endl;
    cudaDeviceSynchronize();
    cudaMemcpy(h_output, d_output, numBytes, cudaMemcpyDeviceToHost);

    cudaEventDestroy(start);
    cudaEventDestroy(end);

    cudaFree(d_output);
    cudaFree(d_input);
    free(h_input);
    return h_output;  
}

bool isMatch(float* a, float* b, int elemCount){
    for (int i = 0; i < elemCount; i++){
        if (fabsf(a[i] - b[i]) > 1e-5) return false;
    }
    return true;
}

int main(){
    int N = 2048, M = 512;
    int numBytes = N * M * sizeof(float);

    float* cpu_output = (float*)malloc(numBytes);    
    float* v2_output1 = (float*)malloc(numBytes);
    float* v2_output2 = (float*)malloc(numBytes);

    cpu_output = transpose_cpu(cpu_output, N, M);
    v2_output1 = v2_time<32, 8>(M, N, 10, 10);
    v2_output2 = v2_time<16, 16>(M, N, 10, 10);

    if (isMatch(cpu_output, v2_output2, N * M)){
        std::cout << "Results Match!" << std::endl;
    }
    else{
        std::cout << "Results not Match!" << std::endl;
    }
}