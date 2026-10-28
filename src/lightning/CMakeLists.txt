cmake_minimum_required(VERSION 3.24)
project(lightning_ops LANGUAGES CXX CUDA)

set(ORT_INCLUDE_DIR "" CACHE PATH "Directory containing ONNX Runtime main C/C++ API headers")
if(NOT EXISTS "${ORT_INCLUDE_DIR}/onnxruntime_cxx_api.h")
  message(FATAL_ERROR "Set ORT_INCLUDE_DIR to the headers from your ONNX Runtime build")
endif()

set(CMAKE_CUDA_ARCHITECTURES 90)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_library(lightning_ops SHARED lightning_ops.cu)
target_include_directories(lightning_ops PRIVATE "${ORT_INCLUDE_DIR}")
target_compile_options(lightning_ops PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:-O3;--fmad=false;-lineinfo>")
set_target_properties(lightning_ops PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
