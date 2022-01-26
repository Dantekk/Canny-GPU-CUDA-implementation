#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

struct Kernel_weights{

      char x[7][7];
      char y[7][7];

};


void init_kernel_weights(Kernel_weights &, unsigned char);
void CannyGPU(unsigned char* img_host,
              unsigned char* out,
              short rows,
              short cols,
              unsigned char kernel_size,
              int low_tr,
              int high_tr,
              unsigned char L2_norm);
