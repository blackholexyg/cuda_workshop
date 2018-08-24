// Share Memory

#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include "timer.h"

__global__
void rgb_to_grey_global(const uchar4* const rgbaImage,
                       unsigned char* const greyImage,
                       int numRows, int numCols)
{
    int idx = threadIdx.x+ blockIdx.x* blockDim.x;
    
    if (idx < numCols*numRows) {
        uchar4 rgba = rgbaImage[idx];
        float grey = .299f * rgba.x + .587f * rgba.y + .114f * rgba.z;
        greyImage[idx] = grey;
    }
}

__global__
void rgb_to_grey_shared(const uchar4* const rgbaImage,
                       unsigned char* const greyImage,
                       int numRows, int numCols)
{
    int idx = threadIdx.x+ blockIdx.x* blockDim.x;
    
    __shared__ uchar4 sh_rgba[256];
    sh_rgba[threadIdx.x] = rgbaImage[idx];
    
    // barrier
    
    
    // __shared__ uchar4 grey[256];
    
    if (idx < numCols*numRows) {
        uchar4 rgba = sh_rgba[threadIdx.x];
        float grey = .299f * rgba.x + .587f * rgba.y + .114f * rgba.z;
        greyImage[idx] = grey;
    }
}

void rgba_to_grey_cuda( const uchar4* const h_rgbaImage,
                        unsigned char *const h_greyImage,
                        size_t numRows, size_t numCols)
{
    uchar4        *d_rgbaImage;
    unsigned char *d_greyImage;
    size_t numPixels= numRows*numCols;
        
    // Alloc memory
    cudaMalloc((void **) &d_rgbaImage,  sizeof(uchar4) * numPixels);
    cudaMalloc((void **) &d_greyImage,  sizeof(uchar4) * numPixels);
    
    // Copy from Host to Device
    cudaMemset(d_greyImage, 0, numPixels * sizeof(unsigned char));
    cudaMemcpy(d_rgbaImage, h_rgbaImage, sizeof(uchar4) * numPixels, cudaMemcpyHostToDevice);
    
    // Kernel Lunch
    const int threadPerBlock = 256;
    const int numBlock = numPixels/threadPerBlock+1;
    
    GpuTimer timer1;
    timer1.Start();
    rgb_to_grey_global<<<numBlock, threadPerBlock>>>(d_rgbaImage, d_greyImage, numRows, numCols);
    timer1.Stop();
    std::cout << "On global memory ran in: " << timer1.Elapsed() << "msecs." << std::endl;
    
/*     GpuTimer timer2;
    timer2.Start();
    rgb_to_grey_shared<<<numBlock, threadPerBlock>>>(d_rgbaImage, d_greyImage, numRows, numCols);
    timer2.Stop();
    std::cout << "On shared memory ran in: " << timer2.Elapsed() << "msecs." << std::endl; */
    
    // Copy from Device to Host
    cudaMemcpy(h_greyImage, d_greyImage, sizeof(unsigned char) * numPixels, cudaMemcpyDeviceToHost);
        
    // Free memory
    cudaDeviceSynchronize(); 
    cudaFree(d_greyImage);
    cudaFree(d_rgbaImage);
}

