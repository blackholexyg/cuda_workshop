// Mutli-dimensional blocks and grids

#include <cuda.h>
#include <cuda_runtime.h>

__global__
void rgb_to_grey_kernel(const uchar4* const rgbaImage,
                       unsigned char* const greyImage,
                       int numRows, int numCols)
{
    int idx = threadIdx.x+ blockIdx.x* blockDim.x;
    int idy = threadIdx.y+ blockIdx.y* blockDim.y;
    
    if (idy<numCols && idx < numRows) {
        int index = numRows*idy + idx;
        uchar4 rgba = rgbaImage[index];
        unsigned char grey = (unsigned char)(0.299f*rgba.x+ 0.587f*rgba.y + 0.114f*rgba.z);
        greyImage[index] = grey;
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
    int blockWidth = 32;
    int blocksX = numRows/blockWidth+1;
    int blocksY = numCols/blockWidth+1;
    
    const dim3 blockSize( blockWidth, blockWidth, 1); 
    const dim3 gridSize( blocksX, blocksY, 1); 
    rgb_to_grey_kernel<<<gridSize, blockSize>>>(d_rgbaImage, d_greyImage, numRows, numCols);
        
    // Copy from Device to Host
    cudaMemcpy(h_greyImage, d_greyImage, sizeof(unsigned char) * numPixels, cudaMemcpyDeviceToHost);
        
    // Free memory
    cudaDeviceSynchronize(); 
    cudaFree(d_greyImage);
    cudaFree(d_rgbaImage);
}

