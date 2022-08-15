// #define TILE_WIDTH 12

// __global__ void TiledSharedMemoryConvolution(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K)
// {
//     int b, m, tx, ty, h_base, w_base, h, w;
//     int X_tile_width = TILE_WIDTH + K - 1;

//     int H_out = H - K + 1;
//     int W_out = H - K + 1;

// #define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
// #define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
// #define w4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

//     extern __shared__ float shmem[];
//     float * X_shared = &shmem[0];
//     float * W_shared = &shmem[X_tile_width * X_tile_width];
//     b = blockIdx.x;
//     m = blockIdx.y;
//     tx = threadIdx.x;
//     ty = threadIdx.y;
//     int w_grid = (W_out-1)/TILE_WIDTH + 1;
//     h_base = (blockIdx.z / w_grid) * TILE_WIDTH;
//     w_base = (blockIdx.z % w_grid) * TILE_WIDTH;
//     h = h_base + ty;
//     w = w_base + tx;
    
//     float sum = 0.0f;
//     int c, p, q;
//     for (c = 0; c < C; c++){
//         if ((tx < K) &&( ty < K)){
//             W_shared[ty * K + tx] = w4d(m, c, ty, tx);
//             // W_shared[tx, ty] = w4d(by, c, tx, ty);
//         }
//         for (int i = h; i < h_base + X_tile_width; i += TILE_WIDTH) {
//             for (int j = w; j < w_base + X_tile_width; j += TILE_WIDTH){
//                 if (i < H && j < W)    
//                     X_shared[(i - h_base) * X_tile_width + (j - w_base)] = x4d(b, c, i, j); // load tile from X[by, c, tx, ty] into shared memory
//                 // X_shared[i - h_base][j - w_base] = x4d(bx, c, h, w);
//                 else
//                     X_shared[(i - h_base) * X_tile_width + (j - w_base)] = 0;
//             }
//         }    
//         __syncthreads();
//         for (p = 0; p < K; p++) {
//             for (q = 0; q < K; q++){ 
//                 if ((tx + p) < X_tile_width && (ty + q) < X_tile_width)    
//                     sum += X_shared[(ty + p)*X_tile_width + (tx + q)] * W_shared[p*K + q];
//                 // sum += X_shared[(h + p)][(w + q)] * W_shared[p][q];
//             }
//         }
//         __syncthreads();
//     }
//     if (h < H_out && w < W_out && b < B && m < M)
//         y4d(b, m, h, w) = sum;
// #undef y4d
// #undef x4d
// #undef w4d
// }