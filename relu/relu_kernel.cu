#include<stdio.h>
#include<iostream>
#include<torch/extension.h>


using namespace std;
__global__ void relu_kernel(float* x, float* y, const int N){
    int id = threadIdx.x + blockDim.x * blockIdx.x;

    if (id < N){
        y[id] = fmaxf(x[id], 0.0f);
    }
}

torch::Tensor relu_cuda(torch::Tensor input){
    auto output = torch::zeros_like(input);

    if (input.options().dtype() != torch::kFloat32){
        cout<<"Tensor info:"<<input.options()<<endl;
        throw runtime_error("value must be kFloat32\n");
    }

    if (!input.is_cuda()){
        throw runtime_error("input tensor must in CUDA device");
    }

    const int ndim = input.dim();
    int N = 1;
    for (int i = 0;i < ndim;i++){
        N *= input.size(i);
    }

    dim3 blocks(256);
    dim3 grids((N + 256 - 1) / 256);

    relu_kernel<<<grids, blocks>>>(input.data_ptr<float>(), output.data_ptr<float>(), N);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("relu", &relu_cuda, "relu");
}