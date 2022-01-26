#include "CudaTimer.cuh"

CudaTimer::CudaTimer(){
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    time=0;
}

CudaTimer::~CudaTimer(){
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void CudaTimer::start_timer(){
    cudaEventRecord(start, 0);
}

void CudaTimer::stop_timer(){
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
}

float CudaTimer::get_time(){

    return time;
}
