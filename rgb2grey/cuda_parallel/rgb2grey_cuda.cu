// CUDA multiple threads

#include <cuda.h>
#include <cuda_runtime.h>

__global__
void rgb2grey_kernel(const uchar4* const rgbaImage,
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

void rgb2grey_cuda( const uchar4* const h_rgbaImage,
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

    // Kernel Launch
    const int threadPerBlock = 512;
    const int numBlock = numPixels/threadPerBlock+1;

    rgb2grey_kernel<<<numBlock, threadPerBlock>>>(d_rgbaImage, d_greyImage, numRows, numCols);

    // Copy from Device to Host
    cudaMemcpy(h_greyImage, d_greyImage, sizeof(unsigned char) * numPixels, cudaMemcpyDeviceToHost);

    // Free memory
    cudaDeviceSynchronize();
    cudaFree(d_greyImage);
    cudaFree(d_rgbaImage);
}
