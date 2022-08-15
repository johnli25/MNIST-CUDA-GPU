#include <cmath>
#include <iostream>
#include "cuda_fp16.h"
#include "gpu-new-forward.h"

#include "new-forward2.cu"
#include "new-forward3.cu"
#include "new-forward4.cu"

using namespace std;
/*
Function paramter definitions:
y - output
x - input
k - kernel
B - batch_size (number of images in x)
M - number of output feature maps
C - number of input feature maps
H - input height dimension
W - input width dimension
K - kernel height and width (K x K)
*/
#define TILE_WIDTH 12
#define TILE_WIDTH2 32
#define CUDA_MAX_NUM_THREADS 1024
#define KERNEL_SIZE 3136

// float **device_y_global; 
// float **device_x_global;

// __global__ void unroll_kernel(float *X_unroll, const float *x, const int B, const int C, const int H, const int W, const int K){
//     const int H_OUT = H - K + 1;
//     const int W_OUT = W - K + 1;
//     int out_dim = H_OUT * W_OUT;

//     // #define x3d(i2, i1, i0) X_unroll[(i2) * (H * W) + (i1) * (W) + i0]
//     #define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
//     // #define x_unroll2d(i1, i0) X_unroll[(i1) * (W) + i0]
//     #define x3d_unroll(i2, i1, i0) X_unroll[(i2)*(w_unroll * C * K * K) + i1 * (w_unroll) + i0]

//     int c, s, h, w, w_unroll, h_base, p, q;
//     int tx = blockIdx.x * blockDim.x + threadIdx.x;
//     // int ty = blockIdx.y * blockDim.y + threadIdx.y;    

//     if (tx < C * out_dim /*&& ty < B*/){
//         c = tx / out_dim;
//         s = tx % out_dim;
//         h = s / W_OUT;
//         w = s % W_OUT;
//         w_unroll = h * W_OUT + w;
//         h_base =  c * K * K;
//         // if (tx > 100 && tx < 105){
//         //     printf("s: %d", s);
//         //     printf("W_out: %d", W_OUT);
//         //     printf("h_output: %d", h);
//         // }
//         for (p = 0; p < K; p++){
//             for (q = 0; q < K; q++){
//                 int h_unroll = h_base + p * K + q;
//                 // x3d_unroll(ty, h_unroll, w_unroll) = x4d(ty, c, h_output + p, w_output + q);
//                 X_unroll[h_unroll * out_dim + w_unroll] = x[(c * H * W) + (h + p) * (W) + w + q];
//             }
//         }
//     }
//     #undef x3d
//     #undef x_unroll2d
// }

// __global__ void matrixMultiplyShared(float *y, const float *x, const float *k, 
//                                     const int B, const int M, 
//                                     const int C, const int H, 
//                                     const int W, const int K){

//   __shared__ float tileM[TILE_WIDTH][TILE_WIDTH];
//   __shared__ float tileN[TILE_WIDTH][TILE_WIDTH];
  
//   int bx = blockIdx.x;
//   int by = blockIdx.y;
  
//   int tx = threadIdx.x;
//   int ty = threadIdx.y;
  
//   int row = by * TILE_WIDTH + ty;
//   int col = bx * TILE_WIDTH + tx;

//   const int H_OUT = H - K + 1;
//   const int W_OUT = W - K + 1;
//   int unroll_size = H_OUT * W_OUT;

//   int numARows = M; //convolution filter W rows = output feature maps
//   int numAColumns = C * K * K;
//   int numBRows = numAColumns; //good thing to note
//   int numBColumns = unroll_size;

//   float output = 0.0;
//   for (int t= 0; t < (numARows-1)/TILE_WIDTH+1; t++){
//     if (row < numARows && t * TILE_WIDTH + tx < numAColumns)
//         tileM[ty][tx] = k[row * numAColumns + t * TILE_WIDTH + tx];
//     else
//         tileM[ty][tx] = 0.0;

//     if(t * TILE_WIDTH + tx < numAColumns && col < numBColumns)
//         tileN[ty][tx] = x[(t * TILE_WIDTH + ty) * numBColumns + col];
//     else
//         tileN[ty][tx] = 0.0;
        
//     __syncthreads();

//     for (int j = 0; j < TILE_WIDTH; j++){
//       output += (tileM[ty][j] * tileN[j][tx]);
//     }
//     __syncthreads();      
//   }
//   if (row < numARows && col < numBColumns)
//     y[row * numBColumns + col] = output;
// }

// __host__ void GPUInterface::conv_forward_gpu(float *device_y, const float *device_x, const float *device_k, const int B, const int M, const int C, const int H, const int W, const int K)
// {
//     // Set the kernel dimensions and call the kernel
//     const int H_out = H - K + 1;
//     const int W_out = W - K + 1;

//     int num_threads = C * H_out * W_out;
//     int num_blocks = ceil((C * H_out * W_out * 1.0) / CUDA_MAX_NUM_THREADS);
//     int output_map_elems = H_out * W_out;
//     int input_map_elems = C * K * K;
//     // dim3 gridDim(ceil(output_map_elems/double(TILE_WIDTH)), ceil(input_map_elems/double(TILE_WIDTH)), 1);
//     // dim3 gridDim(B, M, Z);
//     float * unrolled_device_x;
//     cudaMalloc((void **) &unrolled_device_x, output_map_elems * input_map_elems * sizeof(float));
    // for (int b = 0; b < B; b++){
    //     float * device_x_start = ((float *)device_x) + b * H * K * C;
    //     float * device_y_start = device_y + b * H_out * W_out * M;
    //     unroll_kernel<<<num_blocks, CUDA_MAX_NUM_THREADS>>>(unrolled_device_x, device_x_start, B, C, H, W, K); //1D blocks 

    //     // std::cout << "K: " << K << std::endl;

    //     dim3 gridDim(ceil(H_out*W_out/(1.0 * TILE_WIDTH)), ceil(1.0 * M/(1.0 * TILE_WIDTH)), 1);
    //     dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
    //     matrixMultiplyShared<<<gridDim, blockDim>>>(device_y_start, device_k, unrolled_device_x, B, M, C, H, W, K);
    // }

//     cudaFree(unrolled_device_x);
//     cudaDeviceSynchronize();
// }

// /*GPU forward implementation*/	
// __constant__ float kernel[KERNEL_SIZE];

// __global__ void conv_forward_kernel(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K)
// {
//     /*
//     Modify this function to implement the forward pass described in Chapter 16.
//     We have added an additional dimension to the tensors to support an entire mini-batch
//     The goal here is to be correct AND fast.

//     Function paramter definitions:
//     y - output
//     x - input
//     k - kernel
//     B - batch_size (number of images in x)
//     M - number of output feature maps
//     C - number of input feature maps
//     H - input height dimension
//     W - input width dimension
//     K - kernel height and width (K x K)
//     */

//     const int H_out = H - K + 1;
//     const int W_out = W - K + 1;

//     // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
//     // An example use of these macros:
//     // float a = y4d(0,0,0,0)
//     // y4d(0,0,0,0) = a

// #define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
// #define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
// #define k4d(i3, i2, i1, i0) kernel[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

//     // Insert your GPU convolution kernel code here
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
//             for(p = 0; p < K; p++){
//                 for (q = 0; q < K; q++){
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

__host__ void GPUInterface::conv_forward_gpu_prolog(float *host_y, float *host_x, float *host_k, float **device_y_ptr, float **device_x_ptr, float **device_k_ptr, const int B, const int M, const int C, const int H, const int W, const int K)
{
    // Allocate memory and copy over the relevant data structures to the GPU
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    cudaMalloc((void **) device_x_ptr, B * C * H * W * sizeof(float));
    cudaMalloc((void **) device_k_ptr, M * C * K * K * sizeof(float));
    cudaMalloc((void **) device_y_ptr, B * M * H_out * W_out * sizeof(float));

    // We pass double pointers for you to initialize the relevant device pointers,
    //  which are passed to the other two functions.
    cudaMemcpy(*device_x_ptr, host_x, B * C * H * W * sizeof(float), cudaMemcpyHostToDevice);
    // cudaMemcpy(*device_k_ptr, host_k, M * C * K * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(kernel2, host_k, M * C * K * K * sizeof(float), 0, cudaMemcpyHostToDevice);

    // Useful snippet for error checking
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
        exit(-1);
    }
}

__host__ void GPUInterface::conv_forward_gpu(float * device_y, const float * device_x, const float *device_k, const int B, const int M, const int C, const int H, const int W, const int K)
{
    // Set the kernel dimensions and call the kernel
    // Allocate memory and copy over the relevant data structures to the GPU
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    const int W_GRID = std::ceil(1.0*W_out/TILE_WIDTH);
    const int H_GRID = std::ceil(1.0*H_out/TILE_WIDTH);
    
    const int Z = W_GRID * H_GRID;

    dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 gridDim(B, M, Z);

    restrict_loop_rolling<<<gridDim, blockDim>>>(device_y, device_x, device_k, B, M, C, H, W, K);
    // size_t shmem_size = sizeof(float) * ( (TILE_WIDTH + K-1)*(TILE_WIDTH + K-1) + K*K );
    // TiledSharedMemoryConvolution<<<gridDim, blockDim, shmem_size>>>(device_y, device_x, device_k, B, M, C, H, W, K);
}

__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_y, float *device_y, float *device_x, float *device_k, const int B, const int M, const int C, const int H, const int W, const int K)
{
    // Copy the output back to host
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    cudaMemcpy(host_y, device_y, B*M*H_out*W_out * sizeof(float), cudaMemcpyDeviceToHost); //??
    // Free device memory
    cudaFree(device_y);
    cudaFree(device_x);
    cudaFree(device_k);
}

__host__ void GPUInterface::get_device_properties()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for(int dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
        std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
        std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
        std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
        std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
        std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
        std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
        std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
        std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
    }
}
