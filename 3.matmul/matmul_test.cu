#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

// WMMA tile dimensions are always 16x16x16
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// Threads per block. A 32x8 grid gives 256 threads = 8 warps.
#define THREADS_PER_BLOCK_X 32
#define THREADS_PER_BLOCK_Y 8
#define THREADS_PER_BLOCK (THREADS_PER_BLOCK_X * THREADS_PER_BLOCK_Y)

// We have 8 warps. Let's arrange them in a 4x2 grid logically.
#define WARPS_PER_BLOCK_M 4
#define WARPS_PER_BLOCK_N 2

// The output tile size computed by one thread block
#define BLOCK_DIM_M (WARPS_PER_BLOCK_M * WMMA_M) // 4 * 16 = 64
#define BLOCK_DIM_N (WARPS_PER_BLOCK_N * WMMA_N) // 2 * 16 = 32

// Shared memory K-dimension tile size
#define BLOCK_DIM_K 16

using namespace nvcuda;

// Kernel to perform matrix multiplication using WMMA (Final Corrected Version)
__global__ void wmma_gemm_kernel(const half *a, const half *b, float *d, int M, int N, int K) {
    // --- Correct Warp to Tile Mapping ---
    int warpId = (threadIdx.y * blockDim.x + threadIdx.x) / warpSize;
    int warp_m = warpId / WARPS_PER_BLOCK_N;
    int warp_n = warpId % WARPS_PER_BLOCK_N;
    
    int block_start_m = blockIdx.y * BLOCK_DIM_M;
    int block_start_n = blockIdx.x * BLOCK_DIM_N;

    __shared__ half sh_a[BLOCK_DIM_M][BLOCK_DIM_K];
    __shared__ half sh_b[BLOCK_DIM_K][BLOCK_DIM_N];

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;

    wmma::fill_fragment(acc_frag, 0.0f);

    for (int k_block = 0; k_block < K; k_block += BLOCK_DIM_K) {
        // --- 1. Load tiles from global memory to shared memory ---
        int shmem_idx = threadIdx.y * blockDim.x + threadIdx.x;
        for (int i = shmem_idx; i < BLOCK_DIM_M * BLOCK_DIM_K; i += THREADS_PER_BLOCK) {
            int row = i / BLOCK_DIM_K;
            int col = i % BLOCK_DIM_K;
            if (block_start_m + row < M && k_block + col < K) {
                sh_a[row][col] = a[(block_start_m + row) * K + (k_block + col)];
            } else {
                sh_a[row][col] = __float2half(0.0f); // Zero-padding
            }
        }

        for (int i = shmem_idx; i < BLOCK_DIM_K * BLOCK_DIM_N; i += THREADS_PER_BLOCK) {
            int row = i / BLOCK_DIM_N;
            int col = i % BLOCK_DIM_N;
            if (k_block + row < K && block_start_n + col < N) {
                sh_b[row][col] = b[(k_block + row) * N + (block_start_n + col)];
            } else {
                sh_b[row][col] = __float2half(0.0f); // Zero-padding
            }
        }
        
        __syncthreads();

        // --- 2. Perform matrix multiplication on tiles in shared memory ---
        // Load A tile from shared memory (64x16) to fragment.
        // Stride (ldm) must be the number of columns of the source matrix in shared memory.
        wmma::load_matrix_sync(a_frag, &sh_a[warp_m * WMMA_M][0], BLOCK_DIM_K);
        
        // Load B tile from shared memory (16x32) to fragment.
        // Stride (ldm) must be the number of rows of the source matrix in shared memory.
        wmma::load_matrix_sync(b_frag, &sh_b[0][warp_n * WMMA_N], BLOCK_DIM_K);
        
        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);

        __syncthreads();
    }

    // --- 3. Store the result from fragment to global memory ---
    int output_row = block_start_m + warp_m * WMMA_M;
    int output_col = block_start_n + warp_n * WMMA_N;
    if (output_row < M && output_col < N) {
        // Stride (ldm) for storing must be the number of columns of the DESTINATION matrix in global memory (N).
        wmma::store_matrix_sync(&d[output_row * N + output_col], acc_frag, N, wmma::mem_row_major);
    }
}


// Host code remains the same as the previous version
int main() {
    int M = 512, N = 512, K = 512;

    std::vector<half> h_a(M * K);
    std::vector<half> h_b(K * N);
    std::vector<float> h_d(M * N, 0.0f);

    // Initialize matrices
    for (int i = 0; i < M * K; ++i) h_a[i] = __float2half(1.0f);
    for (int i = 0; i < K * N; ++i) h_b[i] = __float2half(1.0f);

    half *d_a, *d_b;
    float *d_d;
    cudaMalloc(&d_a, sizeof(half) * M * K);
    cudaMalloc(&d_b, sizeof(half) * K * N);
    cudaMalloc(&d_d, sizeof(float) * M * N);

    cudaMemcpy(d_a, h_a.data(), sizeof(half) * M * K, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), sizeof(half) * K * N, cudaMemcpyHostToDevice);
    cudaMemset(d_d, 0, sizeof(float) * M * N);

    dim3 blockDim(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y);
    dim3 gridDim((N + BLOCK_DIM_N - 1) / BLOCK_DIM_N, (M + BLOCK_DIM_M - 1) / BLOCK_DIM_M);

    wmma_gemm_kernel<<<gridDim, blockDim>>>(d_a, d_b, d_d, M, N, K);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error after kernel launch: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }
    cudaDeviceSynchronize();

    cudaMemcpy(h_d.data(), d_d, sizeof(float) * M * N, cudaMemcpyDeviceToHost);

    // Verification
    bool success = true;
    for(int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            if (std::abs(h_d[i * N + j] - static_cast<float>(K)) > 1e-3) {
                std::cout << "Verification failed at (" << i << ", " << j << ")! Got " << h_d[i * N + j] << ", expected " << K << std::endl;
                success = false;
                goto verification_done;
            }
        }
    }
verification_done:
    if (success) {
        std::cout << "WMMA GEMM successful!" << std::endl;
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_d);

    return 0;
}
