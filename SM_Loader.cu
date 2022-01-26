#include "SM_Loader.cuh"

__device__ void SM_data_loader(unsigned char *sm,
                               unsigned char *img,
                               unsigned int sizeSM,
                               unsigned char radius,
                               short rows,
                               short cols){

    // STAGE I

    // Linearize the 2D coordinates of the generic thread -> step from 2D to 1D coordinates.
    int dest = threadIdx.y*blockDim.x+threadIdx.x;

    // Calculate the coordinates of the SM matrix to which the y, x-th thread will have to access.
    int destY = dest/sizeSM;
    int destX = dest%sizeSM;

    // Calculate the shifted coordinates, which each thread in the original img must access.
    int srcY = blockIdx.y*blockDim.y+destY-radius;
    int srcX = blockIdx.x*blockDim.x+destX-radius;
    // Given the 2D coordinates, calculate the 1D index.
    int src = srcY*cols+srcX;

    //
    if (srcY>=0 && srcY<rows && srcX>=0 && srcX<cols)
        sm[destY*sizeSM+destX] = img[src];
    else
        sm[destY*sizeSM+destX] = 0;

    // STAGE II

    for (int iter=1; iter <= (sizeSM*sizeSM)/(blockDim.y*blockDim.x); iter++)
    {
        dest = threadIdx.y * blockDim.x + threadIdx.x + (blockDim.y * blockDim.x); // offset aggiunto
        destY = dest/sizeSM;
        destX = dest%sizeSM;
        srcY = blockIdx.y*blockDim.x+destY-radius;
        srcX = blockIdx.x*blockDim.x+destX-radius;
        src = srcY*cols+srcX;
        if (destY < sizeSM){
            if (srcY >=0 && srcY<rows && srcX>=0 && srcX<cols)
                sm[destY*sizeSM+destX] = img[src];
            else
                sm[destY*sizeSM+destX] = 0;
        }
    }


}
