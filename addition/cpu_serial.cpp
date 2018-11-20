#include <cmath>
#include <cstdio>
#include <ctime>
#include <iostream>

void add(float *a, float *b, float *c, int num)
{
    for (int ii = 0; ii < num; ++ii) {
        c[ii] = a[ii] + b[ii];
    }
}

int main(void)
{
    std::clock_t start_time;
    double duration;

    const int ARR_SIZE =  1000000;
    const int ARR_BYTES =  ARR_SIZE*sizeof(float);

    // Clock start
    start_time = std::clock();

    // Generate input array
    float h_a[ARR_SIZE];
    float h_b[ARR_SIZE];
    float h_c[ARR_SIZE];

    for (int i=0;  i<ARR_SIZE; i++){
        h_a[i] = float(i);
        h_b[i] = float(i)*2.0;
    }

    // Call function add
    add(h_a, h_b, h_c, ARR_SIZE);

    // Clock stop
    duration = ( std::clock() - start_time ) / (double) CLOCKS_PER_SEC;
    std::cout<<"Computing time: "<< duration << "s" << std::endl;

    // Output results
    for(int ii=0; ii<10; ii++){
        std::cout<< h_c[ii]<< ", ";
    }    
    std::cout<< std::endl;

    return 0;
}
