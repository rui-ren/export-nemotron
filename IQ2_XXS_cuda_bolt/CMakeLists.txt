cmake_minimum_required(VERSION 3.22)
set(CMAKE_CUDA_ARCHITECTURES OFF)
project(bolt_native_scale_fixes LANGUAGES CXX CUDA)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

add_subdirectory(compat)
add_subdirectory(iq2-compat)

# The repository's original kernel is unchanged; retain its deployed DSO name.
add_library(bolt_upstream SHARED ../src/bolt_ops.cu)
target_include_directories(bolt_upstream PRIVATE "${ORT_INCLUDE_DIR}")
set_target_properties(bolt_upstream PROPERTIES
  CUDA_STANDARD 17
  POSITION_INDEPENDENT_CODE ON
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
target_compile_options(bolt_upstream PRIVATE
  "$<$<COMPILE_LANGUAGE:CUDA>:-arch=sm_90;-lineinfo>")

include(CTest)
if(BUILD_TESTING)
  set(ORT_LIBRARY "" CACHE FILEPATH "Qualified ORT shared library for integration tests")
  if(ORT_LIBRARY)
    add_library(ort_cuda_runner SHARED tests/ort_cuda_runner.cc)
    target_compile_features(ort_cuda_runner PRIVATE cxx_std_17)
    target_include_directories(ort_cuda_runner PRIVATE "${ORT_INCLUDE_DIR}")
    target_link_libraries(ort_cuda_runner PRIVATE "${ORT_LIBRARY}")
    set_target_properties(ort_cuda_runner PROPERTIES
      LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
  endif()
  foreach(kind IN ITEMS iq2 q8)
    add_executable(test_${kind}_scales tests/scale_precision.cu)
    target_include_directories(test_${kind}_scales PRIVATE "${ORT_INCLUDE_DIR}")
    set_target_properties(test_${kind}_scales PROPERTIES CUDA_STANDARD 17)
    target_compile_options(test_${kind}_scales PRIVATE
      "$<$<COMPILE_LANGUAGE:CUDA>:-arch=sm_90>")
    if(kind STREQUAL "iq2")
      target_compile_definitions(test_${kind}_scales PRIVATE TEST_IQ2=1)
    endif()
    add_test(NAME ${kind}_scale_precision COMMAND test_${kind}_scales)
  endforeach()
endif()
