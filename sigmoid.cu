#include <torch/extension.h>
#include <cuda_runtime.h>
#include<math.h>
#include<iostream>


#define MAX_BLOCK_THREAD 1024
using namespace std;

// CUDA kernel
__global__ void sigmoid_kernel(float* x, float* y, int N){
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < N){
        y[id] = 1.0f / (expf(-x[id]) + 1.0f);
    }
}

torch::Tensor sigmoid_cuda(torch::Tensor input){

    if (input.options().dtype() != torch::kFloat32){
        cout<<"Tensor info:"<<input.options()<<endl;
        throw runtime_error("value must be kFloat32\n");
    }

    if (!input.is_cuda()){
        throw runtime_error("input must on CUDA device\n");
    }


    auto output = torch::zeros_like(input);
    int N = 1;
    for (int i = 0;i < input.dim(); i++){
        N *= input.size(i);
    }

    dim3 blocks(256);
    dim3 grids((N + 256 - 1) / 256);
    sigmoid_kernel<<<grids, blocks>>>(input.data_ptr<float>(), output.data_ptr<float>(), N);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("sigmoid", &sigmoid_cuda, "sigmoid activation (CUDA)");
}