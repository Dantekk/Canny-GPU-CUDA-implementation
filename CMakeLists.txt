### CMakeLists.txt for CUDA and OpenCV

project( main )
cmake_minimum_required(VERSION 2.8)
find_package(CUDA QUIET REQUIRED)
find_package( OpenCV REQUIRED )

# Pass options to NVCC
set(
    CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
    -arch=sm_30
    #--relocatable-device-code true
    )

# Specify include directories
include_directories(
#    kernels
#    utility
    )

# Specify library paths
#link_directories(
#    /opt/foobar/lib
#    /opt/joestuff/lib
#    )

# For compilation ...
# Specify target & source files to compile it from
SET(CUDA_SEPARABLE_COMPILATION ON)
cuda_add_executable(
    main
    main.cpp
    GaussianFilter.cuh
    GaussianFilter.cu
    utility.hpp
    utility.cpp
    CannyGPU.cuh
    CannyGPU.cu
    SM_Loader.cu
    SM_Loader.cuh
    CudaTimer.cu
    CudaTimer.cuh
    )

# For linking ...
target_link_libraries( ${PROJECT_NAME} ${OpenCV_LIBS} )
