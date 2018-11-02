#include <iostream>
#include <string>
#include <cstdio>
#include <ctime>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

#include <cuda_runtime.h>


void rgb2grey_cuda(const uchar4 * const h_rgbaImage,
                   unsigned char* const h_greyImage,
                   size_t numRows, size_t numCols);


void export_image(const std::string& output_file, unsigned char* data_ptr,
                 size_t numRows, size_t numCols) {
  cv::Mat output(numRows, numCols, CV_8UC1, (void*)data_ptr);
  cv::imwrite(output_file.c_str(), output);
}

int main(int argc, char **argv) {
    std::clock_t start_time;
    double duration;

    std::string input_file;
    std::string output_file;
    uchar4        *h_rgbaImage;
    unsigned char *h_greyImage;

    input_file = std::string(argv[1]);

    // Read Image
    cv::Mat image;
    cv::Mat imageRGBA;
    cv::Mat imageGrey;
    image = cv::imread(input_file.c_str(), CV_LOAD_IMAGE_COLOR);
    if (image.empty()) {
        std::cerr << "Couldn't open file: " << input_file << std::endl;
        exit(1);
    }
    cv::cvtColor(image, imageRGBA, CV_BGR2RGBA);
    imageGrey.create(image.rows, image.cols, CV_8UC1);

    size_t numRows = imageRGBA.rows;
    size_t numCols = imageRGBA.cols;

    h_rgbaImage = (uchar4 *)imageRGBA.ptr<unsigned char>(0);
    h_greyImage  = imageGrey.ptr<unsigned char>(0);


    // Converting -- CUDA with shared memory
    start_time = std::clock();

    for(int ii=0; ii<100; ii++){
        rgb2grey_cuda(h_rgbaImage, h_greyImage, numRows, numCols);
    }
    duration = ( std::clock() - start_time ) / (double) CLOCKS_PER_SEC;
    std::cout<<"Running time (CUDA shared memory): "<< duration << "s"
        << std::endl;

    // Output Image
    output_file = "cuda_shared_memory.jpg";
    export_image(output_file, h_greyImage, numRows, numCols);

    return 0;
}
