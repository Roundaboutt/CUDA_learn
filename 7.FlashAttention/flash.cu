#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ 
void forward_kernel(const float* Q, const float* K, const float* V, const int N, const int d,
                    const int Tc, const int Tr, const int Bc, const int Br, const float softmax_scale,
                    float* l, float* m, float* O) {
    const int tid = threadIdx.x;

    const int bx = blockIdx.x;  // batch index
    const int by = blockIdx.y;  // head index

    extern __shared__ float sram[];
    const int tile_size = Bc * d;
    float* Qi = sram;
    float* Kj = &sram[tile_size];
    float* Vj = &sram[2 * tile_size];
    float* S = &sram[3 * tile_size];    // 缓存 QK^T 的结果
    
    const int qkv_offset = (bx * gridDim.y * N * d) + (by * N * d); // gridDim.y = nh
    const int lm_offset = (bx * gridDim.y * N) + (by * N);  // l, m少了d维度
    
    // outer loop
    for (int j = 0; j < Tc; j++){
        // 把 K, V搬到sram
        for (int x = 0; x < d; x++){
            Kj[tid * d + x] = K[qkv_offset + tile_size * j + tid * d + x];
            Vj[tid * d + x] = V[qkv_offset + tile_size * j + tid * d + x];
        } 

        //__syncthreads();      // 这里可以不用同步, 因为一个 block 有32个线程正好是一个warp

        // inner loop
        for (int i = 0; i < Tr; i++){
            // 把 Q 搬到sram
            for (int x = 0; x < d; x++){
                Qi[tid * d + x] = Q[qkv_offset + tile_size * i + tid * d + x];
            }

            float row_m_prev = m[lm_offset + Br * i + tid];
            float row_l_prev = l[lm_offset + Br * i + tid];

            float row_m = -INFINITY;
            // 计算S (Br x Bc) 一个线程只计算一行
            for (int y = 0; y < Bc; y++){
                float sum = 0.f;
                for (int x = 0; x < d; x++){
                    sum += Qi[tid * d + x] * Kj[y * d + x];
                }
                sum *= softmax_scale;
                S[tid * Bc + y] = sum;
                row_m = fmaxf(row_m, sum);
            }

            float row_l = 0;
            // 计算每一行的和
            for (int x = 0; x < Bc; x++){
                S[Bc * tid + x] = __expf(S[Bc * tid + x] - row_m);
                row_l += S[Bc * tid + x];
            }

            // 计算新的 m 和 l
            float row_m_new = fmaxf(row_m_prev, row_m);
            float row_l_new = (__expf(row_m_prev - row_m_new) * row_l_prev) + (__expf(row_m - row_m_new) * row_l);
            
            for (int x = 0; x < d; x++) {
                float pv = 0;  // Pij * Vj
                for (int y = 0; y < Bc; y++) {
                    pv += S[(Bc * tid) + y] * Vj[(y * d) + x];
                }
                O[qkv_offset + (tile_size * i) + (tid * d) + x] = (1 / row_l_new) \
                    * ((row_l_prev * __expf(row_m_prev - row_m_new) * O[qkv_offset + (tile_size * i) + (tid * d) + x]) \
                    + (__expf(row_m - row_m_new) * pv));
            }
            
            m[lm_offset + Br * i + tid] = row_m_new;
            l[lm_offset + Br * i + tid] = row_l_new;
        }
        __syncthreads();
    }
}

torch::Tensor forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    const int Br = 32; const int Bc = 32;
    const int batch = Q.size(0);
    const int N_heads = Q.size(1);
    const int N = Q.size(2);
    const int d = Q.size(3);

    const int Tr = ceil(N / Br); const int Tc = ceil(N / Bc);
    const float softmax_scale = rsqrtf(d);

    auto O = torch::zeros_like(Q);
    auto l = torch::zeros({batch, N_heads, N});
    auto m = torch::full({batch, N_heads, N}, -INFINITY);

    torch::Device device(torch::kCUDA);
    l = l.to(device);
    m = m.to(device);

    const int sram_size = (2 * Br * d * sizeof(float)) + (Bc * d * sizeof(float)) + (Bc * Br * sizeof(float));

    dim3 gridSize(batch, N_heads);
    dim3 blockSize(Bc);

    int max_sram_size;
    cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);
    printf("Max shared memory: %d, requested shared memory: %d \n", max_sram_size, sram_size);

    forward_kernel<<<gridSize, blockSize, sram_size>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
        N, d, Tc, Tr, Bc, Br, softmax_scale,
        l.data_ptr<float>(),
        m.data_ptr<float>(),
        O.data_ptr<float>()
    );
    return O;
}