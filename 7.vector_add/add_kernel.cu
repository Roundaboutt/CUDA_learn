#include<iostream>
#include<cuda_runtime.h>

__global__ void add_kernelv1(float* A, float* B, float* C, const int N)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int global_id = tid + blockDim.x * bid;

    if (global_id >= N) return;

    C[global_id] = A[global_id] + B[global_id];
}



__global__ void add_kernelv2(float* A, float* B, float* C, const int N)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int global_id = tid + bid * blockDim.x;

    constexpr int pack_size = 4;
    const int pack_num = N / pack_size;
    const int pack_off = pack_num * pack_size;


    float4* A_pack = reinterpret_cast<float4*> (A);
    float4* B_pack = reinterpret_cast<float4*> (B);
    float4* C_pack = reinterpret_cast<float4*> (C);

    #pragma unroll
    for (int i = global_id; i < pack_num; i += blockDim.x * gridDim.x)
    {   
        float4 temp_A = A_pack[i];
        float4 temp_B = B_pack[i];
        float4 temp_C;
        
        temp_C.x = temp_A.x + temp_B.x;
        temp_C.y = temp_A.y + temp_B.y;
        temp_C.z = temp_A.z + temp_B.z;
        temp_C.w = temp_A.w + temp_B.w;

        C_pack[i] = temp_C;
    }

    #pragma unroll
    for (int i = global_id + pack_off; i < N; i += blockDim.x * gridDim.x)
    {
        C[i] = A[i] + B[i];
    }
    
}


void add_cpu(float* A, float* B, float* C, const int N)
{
    for (int i = 0; i < N; i++)
    {
        C[i] = A[i] + B[i];
    }
}


int main()
{
    constexpr int N = 1024 * 1024;
    constexpr int numBytes = sizeof(float) * N;

    float* h_A = (float*)malloc(numBytes);
    float* h_B = (float*)malloc(numBytes);
    float* h_C_cpu = (float*)malloc(numBytes);
    float* h_C_gpu = (float*)malloc(numBytes);


    for (int i = 0; i < N; i++)
    {
        h_A[i] = (float)i;
        h_B[i] = (float)i;
    }


    add_cpu(h_A, h_B, h_C_cpu, N);

    float* d_A,* d_B,* d_C;
    cudaMalloc((void**)&d_A, numBytes);
    cudaMalloc((void**)&d_B, numBytes);
    cudaMalloc((void**)&d_C, numBytes);

    cudaMemcpy(d_A, h_A, numBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, numBytes, cudaMemcpyHostToDevice);

    dim3 block(1024);
    dim3 grid((N + block.x - 1) / block.x);

    add_kernelv2<<<grid, block>>>(d_A, d_B, d_C, N);

    cudaMemcpy(h_C_gpu, d_C, numBytes, cudaMemcpyDeviceToHost);


    bool flag = true;
    for (int i = 0; i < N; i++)
    {
        if (abs(h_C_gpu[i] - h_C_cpu[i]) >= 1e-5)
        {   
            flag = false;
            break;
        }
    }

    if (flag)
        std::cout << "right!" << std::endl;
    else
        std::cout << "wrong!" << std::endl;
}