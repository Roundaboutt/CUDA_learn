#include <torch/extension.h>


#define FULLMASK 0xffffffff
#define BLOCKSIZE 1024  
__device__ float BlockReduce(float val){
    const int tid = threadIdx.x;
    const int warpID = tid / warpSize;
    const int laneID = tid % warpSize;

    // warp内归约
    for (int offset = 16; offset > 0; offset >>= 1){
        val += __shfl_down_sync(FULLMASK, val, offset);
    }

    __shared__ float warp_shared[32];
    
    if (laneID == 0)
        warp_shared[warpID] = val;

    __syncthreads();

    if (warpID == 0){
        val = warp_shared[laneID];
        for (int offset = 16; offset > 0; offset >>= 1){
            val += __shfl_down_sync(FULLMASK, val, offset);
        }
    }

    return val;
}

__global__ void reduce_kernel(float* input, float* output, const int N){
    const int global_id = threadIdx.x + blockDim.x * blockIdx.x;
    
    float sum = 0.f;
    for (int i = global_id; i < N; i += gridDim.x * blockDim.x){
        sum += input[i];
    }
    sum = BlockReduce(sum);
    if (threadIdx.x == 0){
        output[blockIdx.x] = sum;
    }
}


torch::Tensor forward(torch::Tensor input){
    int N = 1;
    const int ndim = input.dim();
    for (int i = 0; i < ndim; i++){
        N *= input.size(i);
    }

    auto output = torch::zeros({N});

    int block = BLOCKSIZE;
    int grid = (N + BLOCKSIZE * 2 - 1) / (BLOCKSIZE * 2);
    int n = N;

    while(block > 1){
        reduce_kernel<<<grid, block>>>(input.data_ptr<float>(), output.data_ptr<float>(), n);
        cudaDeviceSynchronize();

        n = block;
        block = (n + BLOCKSIZE * 2 - 1) / (BLOCKSIZE * 2);

        std::swap(input, output);
    }

    reduce_kernel<<<1, block>>>(input.data_ptr<float>(), output.data_ptr<float>(), n);
    return output;

}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("forward", &forward, "forward");
}