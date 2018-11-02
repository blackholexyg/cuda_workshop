cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
project(rgb2grey)

find_package(OpenCV REQUIRED PATHS "/home/yxh109/p20062/software/opencv/3.4.3")
# find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

link_libraries(${OpenCV_LIBS} )
file(GLOB  cpp *.cpp )

CUDA_ADD_EXECUTABLE(rgb2grey ${cpp} rgb2grey_cuda.cu)
