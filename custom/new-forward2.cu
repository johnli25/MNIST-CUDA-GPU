#define TILE_WIDTH 12
#define TILE_WIDTH2 32
#define CUDA_MAX_NUM_THREADS 1024
#define KERNEL_SIZE 3136

__constant__ float kernel2[KERNEL_SIZE];

__global__ void restrict_loop_rolling(float *y, const float * __restrict__ x, const float * __restrict__ k, 
    const int B, const int M, 
    const int C, const int H, 
    const int W, const int K){

#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) kernel2[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

    int n, m, c, h, w;
    int w_grid = (W_out-1)/TILE_WIDTH + 1;
    int h_grid = (H_out-1)/TILE_WIDTH + 1;
    int z = w_grid * h_grid;
    n = blockIdx.x;
    m = blockIdx.y;
    h = (blockIdx.z / w_grid)*TILE_WIDTH + threadIdx.y;
    w = (blockIdx.z % w_grid)*TILE_WIDTH + threadIdx.x;
    float sum = 0.0f;
    if (h < H_out && w < W_out){
        for (c = 0; c < C; c++){
            sum += x4d(n, c, h + 0, w + 0) * k4d(m, c, 0, 0);
            sum += x4d(n, c, h + 0, w + 1) * k4d(m, c, 0, 1);
            sum += x4d(n, c, h + 0, w + 2) * k4d(m, c, 0, 2);
            sum += x4d(n, c, h + 0, w + 3) * k4d(m, c, 0, 3);
            sum += x4d(n, c, h + 0, w + 4) * k4d(m, c, 0, 4);
            sum += x4d(n, c, h + 0, w + 5) * k4d(m, c, 0, 5);
            sum += x4d(n, c, h + 0, w + 6) * k4d(m, c, 0, 6);

            sum += x4d(n, c, h + 1, w + 0) * k4d(m, c, 1, 0);
            sum += x4d(n, c, h + 1, w + 1) * k4d(m, c, 1, 1);
            sum += x4d(n, c, h + 1, w + 2) * k4d(m, c, 1, 2);
            sum += x4d(n, c, h + 1, w + 3) * k4d(m, c, 1, 3);
            sum += x4d(n, c, h + 1, w + 4) * k4d(m, c, 1, 4);
            sum += x4d(n, c, h + 1, w + 5) * k4d(m, c, 1, 5);
            sum += x4d(n, c, h + 1, w + 6) * k4d(m, c, 1, 6);

            sum += x4d(n, c, h + 2, w + 0) * k4d(m, c, 2, 0);
            sum += x4d(n, c, h + 2, w + 1) * k4d(m, c, 2, 1);
            sum += x4d(n, c, h + 2, w + 2) * k4d(m, c, 2, 2);
            sum += x4d(n, c, h + 2, w + 3) * k4d(m, c, 2, 3);
            sum += x4d(n, c, h + 2, w + 4) * k4d(m, c, 2, 4);
            sum += x4d(n, c, h + 2, w + 5) * k4d(m, c, 2, 5);
            sum += x4d(n, c, h + 2, w + 6) * k4d(m, c, 2, 6);

            sum += x4d(n, c, h + 3, w + 0) * k4d(m, c, 3, 0);
            sum += x4d(n, c, h + 3, w + 1) * k4d(m, c, 3, 1);
            sum += x4d(n, c, h + 3, w + 2) * k4d(m, c, 3, 2);
            sum += x4d(n, c, h + 3, w + 3) * k4d(m, c, 3, 3);
            sum += x4d(n, c, h + 3, w + 4) * k4d(m, c, 3, 4);
            sum += x4d(n, c, h + 3, w + 5) * k4d(m, c, 3, 5);
            sum += x4d(n, c, h + 3, w + 6) * k4d(m, c, 3, 6);

            sum += x4d(n, c, h + 4, w + 0) * k4d(m, c, 4, 0);
            sum += x4d(n, c, h + 4, w + 1) * k4d(m, c, 4, 1);
            sum += x4d(n, c, h + 4, w + 2) * k4d(m, c, 4, 2);
            sum += x4d(n, c, h + 4, w + 3) * k4d(m, c, 4, 3);
            sum += x4d(n, c, h + 4, w + 4) * k4d(m, c, 4, 4);
            sum += x4d(n, c, h + 4, w + 5) * k4d(m, c, 4, 5);
            sum += x4d(n, c, h + 4, w + 6) * k4d(m, c, 4, 6);

            sum += x4d(n, c, h + 5, w + 0) * k4d(m, c, 5, 0);
            sum += x4d(n, c, h + 5, w + 1) * k4d(m, c, 5, 1);
            sum += x4d(n, c, h + 5, w + 2) * k4d(m, c, 5, 2);
            sum += x4d(n, c, h + 5, w + 3) * k4d(m, c, 5, 3);
            sum += x4d(n, c, h + 5, w + 4) * k4d(m, c, 5, 4);
            sum += x4d(n, c, h + 5, w + 5) * k4d(m, c, 5, 5);
            sum += x4d(n, c, h + 5, w + 6) * k4d(m, c, 5, 6);

            sum += x4d(n, c, h + 6, w + 0) * k4d(m, c, 6, 0);
            sum += x4d(n, c, h + 6, w + 1) * k4d(m, c, 6, 1);
            sum += x4d(n, c, h + 6, w + 2) * k4d(m, c, 6, 2);
            sum += x4d(n, c, h + 6, w + 3) * k4d(m, c, 6, 3);
            sum += x4d(n, c, h + 6, w + 4) * k4d(m, c, 6, 4);
            sum += x4d(n, c, h + 6, w + 5) * k4d(m, c, 6, 5);
            sum += x4d(n, c, h + 6, w + 6) * k4d(m, c, 6, 6);
        }
        y4d(n, m, h, w) = sum;
    }
#undef y4d
#undef x4d
#undef k4d
}

// __global__ void restrict_constant_loop_size(float *y, const float * __restrict__ x, const float * __restrict__ k, 
//     const int B, const int M, 
//     const int C, const int H, 
//     const int W, const int K){

// #define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
// #define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
// #define k4d(i3, i2, i1, i0) kernel2[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

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
//     float sum = 0.0f;
//     if (h < H_out && w < W_out){
//         for (c = 0; c < C; c++){
//             for(p = 0; p < 7; p++){
//                 for (q = 0; q < 7; q++){
//                     sum += x4d(n, c, h+p, w+q) * k4d(m, c, p, q);
//                 }
//             }
//         }
//         y4d(n, m, h, w) = sum;
//     }
// #undef y4d
// #undef x4d
// #undef k4d
// }