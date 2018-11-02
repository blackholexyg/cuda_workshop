#include <cstdio>
#include <cmath>

#define ARR_SIZE 64

__global__
void shift_no_barrier(int *d_array)
{
    int idx=threadIdx.x;
    __shared__ int sh_array[ARR_SIZE];
    sh_array[idx]=threadIdx.x;
    if (idx<ARR_SIZE-1)
        sh_array[idx] = sh_array[idx+1];
    d_array[idx] = sh_array[idx];
}


__global__
void shift_with_barrier(int *d_array)
{
    int idx=threadIdx.x;
    __shared__ int sh_array[ARR_SIZE];
    sh_array[idx]=threadIdx.x;
    __syncthreads();
    if (idx<ARR_SIZE-1) {
        int temp = sh_array[idx+1];
        __syncthreads();
        sh_array[idx] = temp;
        __syncthreads();
    }
    d_array[idx] = sh_array[idx];
}

int main(void)
{
    const int ARR_BYTES=ARR_SIZE*sizeof(float);
    int h_array[ARR_SIZE];

    // Declare and alloc array on device
    int *d_array;
    cudaMalloc((void **) &d_array, ARR_BYTES);

    // No Barrier
    shift_no_barrier<<<1, ARR_SIZE>>>(d_array);
    cudaMemcpy(h_array, d_array, ARR_BYTES, cudaMemcpyDeviceToHost);
    for (int i=0;  i<ARR_SIZE; i++){
        printf("%d, ", h_array[i]);
    }
    printf("\n");

    // With Barrier
    shift_with_barrier<<<1, ARR_SIZE>>>(d_array);
    cudaMemcpy(h_array, d_array, ARR_BYTES, cudaMemcpyDeviceToHost);
    for (int i=0;  i<ARR_SIZE; i++){
        printf("%d, ", h_array[i]);
    }
    printf("\n");

    // Free memory
    cudaFree(d_array);

    return 0;
}
