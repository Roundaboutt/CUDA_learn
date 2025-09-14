#include<stdio.h>
#include<iostream>
#include<torch/extension.h>


#define TILE_SIZE 32
__global__ void mm_kernel(float* A,float* B, int width){
    
}