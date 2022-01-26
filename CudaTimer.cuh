#include <cuda.h>
#include <cuda_runtime.h>

/*
    Class for simple timing management using CUDA routines.

    ** start_timer() allows you to start the timer
    ** stop_timer() allows you to stop the timer
    ** get_time() returns the time recorded between start_timer() and stop_timer()
*/
class CudaTimer{

    public:
      CudaTimer();
      ~CudaTimer();
      void start_timer();
      void stop_timer();
      float get_time();

    private:
      float time;
      cudaEvent_t start, stop;

};
