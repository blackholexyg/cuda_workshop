#include <iostream>
#include <string>
#include <cstdio>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

#include <cuda_runtime.h>

void rgba_to_grey_serial(const uchar4 * const h_rgbaImage, 
                        unsigned char* const h_greyImage, 
                        size_t numRows, size_t numCols);

void rgba_to_grey_cuda(const uchar4 * const h_rgbaImage, 
                        unsigned char* const h_greyImage, 
                        size_t numRows, size_t numCols);


void export_image(const std::string& output_file, unsigned char* data_ptr,
                 size_t numRows, size_t numCols) {
  cv::Mat output(numRows, numCols, CV_8UC1, (void*)data_ptr);
  cv::imwrite(output_file.c_str(), output);
}
       
int main(int argc, char **argv) {
    
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
    
    // Converting -- serial version
    rgba_to_grey_serial(h_rgbaImage, h_greyImage, numRows, numCols);
    
    output_file = "output-serial.jpg";
    export_image(output_file, h_greyImage, numRows, numCols);
    
    // Converting -- CUDA version
    rgba_to_grey_cuda(h_rgbaImage, h_greyImage, numRows, numCols);
    
    output_file = "output-cuda.jpg";
    export_image(output_file, h_greyImage, numRows, numCols);
    
    return 0;
}

