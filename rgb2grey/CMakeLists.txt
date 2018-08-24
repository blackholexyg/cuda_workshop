cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
project(rgb2grey)

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

link_libraries(${OpenCV_LIBS} )

# set (EXECUTABLE_OUTPUT_PATH "${CMAKE_BINARY_DIR}/bin/")

file( GLOB  cpp *.cpp )

CUDA_ADD_EXECUTABLE(rgb2grey ${cpp} rgb2grey_cuda_v4.cu)