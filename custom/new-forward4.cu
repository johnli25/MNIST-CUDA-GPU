// #include "cuda_fp16.h"
// #define TILE_WIDTH 12
// #define KERNEL_SIZE 3136

// __constant__ float kernel_fp16[KERNEL_SIZE];

// __global__ void FP16_restrict_constant_loop_size(float *y, const float * x, const float * k, 
//     const int B, const int M, 
//     const int C, const int H, 
//     const int W, const int K){

// #define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
// #define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
// #define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

//     const int H_out = H - K + 1;
//     const int W_out = W - K + 1;

//     int n, m, c, h, w, p, q;
//     int w_grid = (W_out-1)/TILE_WIDTH + 1;
//     int h_grid = (H_out-1)/TILE_WIDTH + 1;
//     int z = w_grid * h_grid;
//     n = blockIdx.x;
//     m = blockIdx.y;
//     h = (blockIdx.z / w_grid)*TILE_WIDTH + threadIdx.y;
//     w = (blockIdx.z % w_grid)*TILE_WIDTH + threadIdx.x;
//     __half sum = 0.0f;
//     if (h < H_out && w < W_out){
//         for (c = 0; c < C; c++){
//             for(p = 0; p < 7; p++){
//                 for (q = 0; q < 7; q++){
//                     sum += __float2half(x4d(n, c, h+p, w+q)) * __float2half(k4d(m, c, p, q));
//                 }
//             }
//         }
//         y4d(n, m, h, w) = sum;
//     }
// #undef y4d
// #undef x4d
// #undef k4d
// }