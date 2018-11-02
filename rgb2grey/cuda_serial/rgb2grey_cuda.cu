// CUDA single thread

#include <cuda.h>
#include <cuda_runtime.h>

__global__
void rgb2grey_kernel(const uchar4* const rgbaImage,
                       unsigned char* const greyImage,
                       int numRows, int numCols)
{
    for (size_t r = 0; r < numRows; ++r) {
        for (size_t c = 0; c < numCols; ++c) {
            uchar4 rgba = rgbaImage[r * numCols + c];
            float channelSum = .299f * rgba.x + .587f * rgba.y + .114f * rgba.z;
            greyImage[r * numCols + c] = channelSum;
        }
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
    rgb2grey_kernel<<<1, 1>>>(d_rgbaImage, d_greyImage, numRows, numCols);

    // Copy from Device to Host
    cudaMemcpy(h_greyImage, d_greyImage, sizeof(unsigned char) * numPixels, cudaMemcpyDeviceToHost);

    // Free memory
    cudaDeviceSynchronize();
    cudaFree(d_greyImage);
    cudaFree(d_rgbaImage);
}
