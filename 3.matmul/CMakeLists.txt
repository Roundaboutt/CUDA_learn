add_executable(matmul_v1 matmul_v1.cu)
add_executable(matmul_v2 matmul_v2.cu)
add_executable(matmul_v3 matmul_v3.cu)
add_executable(matmul_test matmul_test.cu)

target_link_libraries(matmul_v1 PRIVATE CUDA::cublas)
target_link_libraries(matmul_v2 PRIVATE CUDA::cublas)
target_link_libraries(matmul_v3 PRIVATE CUDA::cublas)
