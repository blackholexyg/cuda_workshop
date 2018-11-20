#include <cmath>
#include <cstdio>
#include <ctime>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>

__global__ void sum_shared_mem(float *array)
{
    int idx = threadIdx.x;
    float sum=0.0f;

    __shared__ float sh_array[512];
    sh_array[idx] = array[idx];
    
    __syncthreads();

    for (int i=0; i<idx; i++){
        sum+= sh_array[i];
    }

    __syncthreads();

    array[idx] = sum;
}


__global__ void sum_global_mem(float *array)
{
    int idx = threadIdx.x;
    float sum=0.0f;

    for (int i=0; i<idx; i++){
        sum+= array[i];
    }
    __syncthreads();
    array[idx] = sum;
}


int main(void)
{
    std::clock_t start_time;
    double duration;

    const int ARR_BYTES =  512*sizeof(float);

    // Clock start
    start_time = std::clock();
    
    // Declare and alloc array on host
    float h_array[512];

    // initialize input array 
    for (int i=0;  i<512; i++){
        h_array[i] = float(i);
    }

    // Declare and alloc array on device
    float *d_array;
    cudaMalloc((void **) &d_array, ARR_BYTES);

    // Transfer to device
    cudaMemcpy(d_array, h_array, ARR_BYTES, cudaMemcpyHostToDevice);

    // Call kernel function
    sum_shared_mem<<<1, 512>>>(d_array);

    // Transfer results to host
    cudaMemcpy(h_array, d_array, ARR_BYTES, cudaMemcpyDeviceToHost);

    // Clock stop
    duration = ( std::clock() - start_time ) / (double) CLOCKS_PER_SEC;
    std::cout<<"Computing time: "<< duration << "s" << std::endl;

    // Output results
    for(int ii=0; ii<10; ii++){
        std::cout<< h_array[ii]<< ", ";
    }    
    std::cout<< std::endl;

    return 0;
}
