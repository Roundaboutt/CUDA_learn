#include <stdio.h>
#include <cuda_runtime.h>

__global__ void non_atomic_add(int* data) {
    int temp = *data;
    temp = temp + 1;
    *data = temp;
}

__global__ void atomic_add(int* data){
    atomicAdd(data, 1);
}

int main() {
    int *d_data1,* d_data2;
    int h_data1 = 0, h_data2 = 0;

    cudaMalloc(&d_data1, sizeof(int));
    cudaMalloc(&d_data2, sizeof(int));
    cudaMemcpy(d_data1, &h_data1, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_data2, &h_data1, sizeof(int), cudaMemcpyHostToDevice);

    non_atomic_add<<<1024, 256>>>(d_data1);
    atomic_add<<<1024, 256>>>(d_data2);

    cudaMemcpy(&h_data1, d_data1, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_data2, d_data2, sizeof(int), cudaMemcpyDeviceToHost);

    printf("Final value: %d (expected %d if no race condition)\n", h_data1, 1024 * 256);
    printf("Final value: %d (expected %d if race condition)\n", h_data2, 1024 * 256);

    cudaFree(d_data1);
    cudaFree(d_data2);
    return 0;
}
