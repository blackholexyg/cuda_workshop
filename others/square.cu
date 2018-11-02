#include <cstdio>
#include <math.h>

// Kernel function to square a float
__global__
void square(float *d_out, float *d_in)
{
    int idx = threadIdx.x; // Get thread ID
    float f = d_in[idx];
    d_out[idx] = f*f;
}

int main(void)
{
    const int ARR_SIZE =  64;
    const int ARR_BYTES =  ARR_SIZE*sizeof(float);

    // Generate input array
    float h_in[ARR_SIZE];
    for (int i=0;  i<ARR_SIZE; i++){
        h_in[i] = float(i);
    }
    float h_out[ARR_SIZE];

    // Declare and alloc array on device
    float *d_in;
    float *d_out;
    cudaMalloc((void **) &d_in, ARR_BYTES);
    cudaMalloc((void **) &d_out, ARR_BYTES);

    // Transfer to device
    cudaMemcpy(d_in, h_in, ARR_BYTES, cudaMemcpyHostToDevice);

    // Run kernel on 64 elements on the GPU
    square<<<1, ARR_SIZE>>>(d_out, d_in);

    // Transfer results to host
    cudaMemcpy(h_out, d_out, ARR_BYTES, cudaMemcpyDeviceToHost);

	//  Output
    for (int i=0;  i<ARR_SIZE; i++){
        printf("%f, ", h_out[i]);
    }
    printf("\n");

    // Free memory
    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}
