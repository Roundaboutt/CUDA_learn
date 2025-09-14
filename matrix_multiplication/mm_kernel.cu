#include<stdio.h>
#include<torch/extension.h>


using namespace std;
__global__ void mm_kernel(float* M, float* N, float* P,const int width){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < width && col < width){
        float Pvalue = 0;
        for (int i = 0;i < width; i++){
            // M[row, i] N[i, col] M,N都是二维矩阵一维化之后的数组
            Pvalue += M[width * row + i] * N[width * i + col];
        }

        P[row * width + col] = Pvalue;
    }
}

torch::Tensor mm_cuda(torch::Tensor X, torch::Tensor Y){
    if (X.options().dtype() != torch::kFloat32){
        throw runtime_error("value must be kFloat32\n");
    }
    if (Y.options().dtype() != torch::kFloat32){
        throw runtime_error("value must be kFloat32\n");
    }
    
    if (!X.is_cuda()){
        throw runtime_error("tensor must be in CUDA device\n");
    }
    if (!Y.is_cuda()){
        throw runtime_error("tensor must be in CUDA device\n");
    } 
    
    const int width = X.size(0);    // 假设是方阵
    auto res = torch::zeros_like(X);


    dim3 blocks(32, 32);
    dim3 grids((width + 32 - 1) / 32, (width + 32 - 1) / 32);

    mm_kernel<<<grids, blocks>>>(
        X.data_ptr<float>(),
        Y.data_ptr<float>(),
        res.data_ptr<float>(),
        width
    );
    return res;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("mm", &mm_cuda, "mm");
}