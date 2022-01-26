#include <cuda.h>

__device__ void SM_data_loader(unsigned char *sm,
                               unsigned char *img,
                               unsigned int sizeSM,
                               unsigned char radius,
                               short rows,
                               short cols);
