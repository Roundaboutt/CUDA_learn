#include<iostream>
#include"cuda_runtime.h"

void rmsnorm_cpu(float* input, float* output, int batch, int size, float* weight, float eps){
    for (int i = 0; i < batch; i++){
        float* input_start = input + size * i;
        float* output_start = output + size * i;

        float sum = 0.f;
        for (int j = 0; j < size; j++){
            sum += output_start[j] * output_start[j];
        }

        float rms = 1.f / std::sqrt(sum / static_cast<float>(size) + eps);

        for (int j = 0; j < size; j++){
            output_start[j] = input_start[j] * weight[j] * rms;
        }
    }
}

