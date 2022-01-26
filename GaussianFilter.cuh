#include <cuda.h>

void GaussianFilterGPU(unsigned char* img_host,
                       unsigned char* out,
                       unsigned char kernel_size,
                       double sigma,
                       short rows,
                       short cols);

struct Kernel_Gaussian_weights{

      double weights[9][9];
};
