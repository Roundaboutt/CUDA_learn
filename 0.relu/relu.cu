#include<cuda_runtime.h>
#include <torch/extension.h>

__global__ void relu_naive(float* input, float* output, int N){
    const int global_id = threadIdx.x + blockIdx.x * blockDim.x;
    if (global_id >= N) return;
    input[global_id] = output[global_id] > 0 ? output[global_id]:0;
}


#define FETCH_FLOAT4(ptr) (reinterpret_cast<float4*>(&(ptr))[0])
__global__ void relu_float4(float* input, float* output, int N){
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int global_id = (tid + bid * blockDim.x) * 4;
    if (global_id >= N) return;
    float4 temp_in = FETCH_FLOAT4(input[global_id]);
    float4 temp_out;
    
    temp_out.x = temp_in.x > 0 ? temp_in.x : 0;
    temp_out.y = temp_in.y > 0 ? temp_in.y : 0;
    temp_out.z = temp_in.z > 0 ? temp_in.z : 0;
    temp_out.w = temp_in.w > 0 ? temp_in.w : 0;
    FETCH_FLOAT4(output[global_id]) = temp_out;
}

torch::Tensor forward(torch::Tensor input){
    
    torch::Tensor output = torch::zeros_like(input);

    const int ndim = input.dim();
    int N = 1;
    for (int i = 0; i < ndim; i++){
        N *= input.size(i);
    }
    dim3 block(256 / 4);
    dim3 grid((N + 256 - 1) / 256);
    relu_float4<<<grid, block>>>(input.data_ptr<float>(), output.data_ptr<float>(), N);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("forward", &forward, "forward");
}