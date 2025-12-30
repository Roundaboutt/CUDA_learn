// #include"cuda_runtime.h"

// template <typename Dtype>
// __global__ void im2col_gpu_kernel(const int n, const Dtype* data_im,
//                                   const int height, const int width,
//                                   const int kernel_h, const int kernel_w,
//                                   const int height_col, const int width_col,
//                                   Dtype* data_col) {
//     CUDA_KERNEL_LOOP(index, n); {
//         const int h_index = index / width_col;
//         const int h_col = h_index % height_col;
//         const int w_col = index % width_col;
//         const int c_im = h_index / height_col;
//         const int c_col = c_im * kernel_h * kernel_w;
//         // 卷积窗口的行列位置
//         const int h_offset = h_col;  
//         const int w_offset = w_col;
//         // 定位到展开后输出的中行列位置
//         Dtype* data_col_ptr = data_col + (c_col * height_col + h_col) * width_col + w_col;
//         const Dtype* data_im_ptr = data_im + (c_im * height + h_offset) * width + w_offset;

//         for (int i = 0; i < kernel_h; ++i) {
//             for (int j = 0; j < kernel_w; ++j) {
//                 // 在行列位置中进行滑动
//                 int h_im = h_offset + i;
//                 int w_im = w_offset + j;
//                 *data_col_ptr = (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width)
//                               ? data_im_ptr[i * width + j]
//                               : 0;
//                 // 展开后是按列存放的，所以转到下一行，如上图所示
//                 data_col_ptr += height_col * width_col;
//             }
//         }
//     }
// }

int main()
{
    
}