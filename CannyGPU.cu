#include "CannyGPU.cuh"
#include "SM_Loader.cuh"

// CUDA global constants
__constant__ Kernel_weights const_weights;

__global__ void sobel_kernel(unsigned char* img, unsigned char* sobel_module, float* sobel_dir, unsigned char kernel_size, unsigned char radius,
                             unsigned int sizeSM, short rows, short cols, unsigned char L2_norm);

__global__ void non_max_suppresion_kernel(unsigned char* sobel_module, float* sobel_dir, unsigned char* out,
                                          unsigned int sizeSM, unsigned char radius, short rows, short cols);

__global__ void hysteresis_kernel(unsigned char* img_non_max_sup, unsigned char* out, unsigned char* sobel_module,
                                  unsigned int sizeSM, unsigned char radius, short rows, short cols, int low_tr, int high_tr);

void CannyGPU(unsigned char* img_host,
              unsigned char* out,
              short rows,
              short cols,
              unsigned char kernel_size,
              int low_tr,
              int high_tr,
              unsigned char L2_norm){

    unsigned char *out_non_max_device, *out_device;
    float *sobel_dir_device;
    unsigned char *img_device, *sobel_module_device;
    unsigned int factor=16;
    unsigned int size;
    int sizeSMbyte;
    unsigned char radius;


    dim3 num_blocks, num_threads_per_block;
    // Kernel config.
    num_threads_per_block.y=factor;
    num_threads_per_block.x=factor;
    //
    num_blocks.y = rows/num_threads_per_block.y+((rows%num_threads_per_block.y)==0? 0:1);
    num_blocks.x = cols/num_threads_per_block.x+((cols%num_threads_per_block.x)==0? 0:1);

    // image allocation size.
    size=rows*cols*sizeof(unsigned char);

    // Data allocation on device.
    cudaMalloc((void**)&img_device, size);
    cudaMalloc((void**)&sobel_dir_device, rows*cols*sizeof(float));
    cudaMalloc((void**)&sobel_module_device, size);
    cudaMalloc((void**)&out_non_max_device, size);
    cudaMalloc((void**)&out_device, size);

    // Copy data from host to device.
    cudaMemcpy(img_device, img_host, size, cudaMemcpyHostToDevice);

    // Defines the kernel weights.
    Kernel_weights k;
    init_kernel_weights(k, kernel_size);
    // Copy the struct kernel weights to Constant Memory.
    cudaMemcpyToSymbol(const_weights, &k, sizeof(k));

    // Calculate the radius of convolution radius.
    radius=int(floor((kernel_size-1)/2));

    // Calculate the size of the SM needed.
    sizeSMbyte = (num_threads_per_block.y+kernel_size-1)*(num_threads_per_block.x+kernel_size-1)*sizeof(unsigned char);
    unsigned int sizeSM = (num_threads_per_block.x+kernel_size-1);


    // Sobel kernel
    sobel_kernel<<<num_blocks, num_threads_per_block, sizeSMbyte>>>(img_device, sobel_module_device, sobel_dir_device, kernel_size, radius, sizeSM, rows, cols, L2_norm);
    cudaDeviceSynchronize();

    // NMS kernel
    sizeSMbyte = (num_threads_per_block.x+3-1)*(num_threads_per_block.y+3-1)*sizeof(unsigned char);
    sizeSM = (num_threads_per_block.x+3-1); // per questa fase viene utilizzato sempre un kernel 3x3
    radius=1; // kernel 3x3 -> quindi radius=1
    non_max_suppresion_kernel<<<num_blocks, num_threads_per_block, sizeSMbyte>>>(sobel_module_device, sobel_dir_device, out_non_max_device, sizeSM, radius, rows, cols);
    cudaDeviceSynchronize();
    // Hysteresis kernel
    sizeSMbyte = 2*(num_threads_per_block.x+3-1)*(num_threads_per_block.y+3-1)*sizeof(unsigned char);
    hysteresis_kernel<<<num_blocks, num_threads_per_block, sizeSMbyte>>>(out_non_max_device, out_device, sobel_module_device, sizeSM, radius, rows, cols, low_tr, high_tr);
    cudaDeviceSynchronize();

    // Copy the resulting array from the device to the host.
    cudaMemcpy(out, out_device, size, cudaMemcpyDeviceToHost);

    // Free the memory from the device.
    cudaFree(out_non_max_device);
    cudaFree(out_device);
    cudaFree(img_device);
    cudaFree(sobel_module_device);
    cudaFree(sobel_dir_device);

}


__global__ void hysteresis_kernel(unsigned char* img_non_max_sup,
                                  unsigned char* out,
                                  unsigned char* sobel_module,
                                  unsigned int sizeSM,
                                  unsigned char radius,
                                  short rows,
                                  short cols,
                                  int low_tr,
                                  int high_tr){

    /*
    The SM is a contiguous memory area, so if you need to allocate multiple arrays ...
    you have to allocate them in a contiguous manner.
    For example, if I need to allocate two arrays; after allocating the first,
    just define a pointer to the memory location immediately following the memory address
    of the last element of the first array, and start allocating from that location onwards.
    */
    extern __shared__ unsigned char sm[];
    unsigned char* sm_mag = &sm[0];
    unsigned char* sm_non_max = &sm[sizeSM*sizeSM];

    // Load the magnitude of the pixels that fall into the block from GM to SM.
    SM_data_loader(sm_mag, sobel_module, sizeSM, radius, rows, cols);
    // Load the value of the pixels of the matrix resulting from the NMS phase that fall into the block from GM to SM.
    SM_data_loader(sm_non_max, img_non_max_sup, sizeSM, radius, rows, cols);

    // Synchronize all threads in the block to make sure all threads have finished writing within the SM.
    __syncthreads();

    // Compute the global indexes of the thread.
    unsigned int y = blockIdx.y*blockDim.y+threadIdx.y;
    unsigned int x = blockIdx.x*blockDim.x+threadIdx.x;

    // if the thread "falls" inside the image
    if(y<rows && x<cols){

        ////
        // I remember that you have to add a shift of + radius to the x and y coordinates when accessing the SM ...
        // to make sure that the thread index inside the block corresponds to the pixel index inside the SM matrix.
        ////
        int local_index = (threadIdx.y+radius)*sizeSM+(threadIdx.x+radius);

        // if in the previous phase the pixel has been discarded, it cannot be "imaged",
        // and therefore 0 is assigned in the final matrix.
        if(sm_non_max[local_index]==0) out[y*cols+x]=0;
        // The pixels that are edges -> coming from the NMS phase are considered.
        else{

            bool edge = false;
            // If the magnitude of the pixel is greater than the high tr, then it is a "strong" edge.
            if(sm_mag[local_index]>high_tr) edge=true;
            // If the pixel magnitude is less than the low tr, then the edge is discarded.
            else if(sm_mag[local_index]<low_tr) edge=false;
            // If the magnitude of the pixel is between the two thresholds,
            // then it is considered as a valid edge only if it is in an 8-connected neighborhood of a "strong" edge.
            else if(sm_mag[local_index]>=low_tr && sm_mag[local_index]<=high_tr){

                for(int i=0; i<3; i++)
                    for(int j=0; j<3; j++){

                        // If a pixel that is in a neighborhood 8-connected to
                        // the considered "eligible" edge pixel is a "strong" edge, then it is also considered as a valid edge pixel.
                        if(sm_mag[(threadIdx.y+i)*sizeSM+(threadIdx.x+j)]>high_tr){
                              edge=true;
                              // Trick to get out of the double for loop
                              i=j=3;
                          }
                    }
            }

            // Mark whether it is an edge pixel or not.
            if(edge) out[y*cols+x]=255;
              else out[y*cols+x]=0;

        }
    }
}

__global__ void non_max_suppresion_kernel(unsigned char* sobel_module,
                                          float* sobel_dir,
                                          unsigned char* out,
                                          unsigned int sizeSM,
                                          unsigned char radius,
                                          short rows,
                                          short cols){

    // Define shared memory data
    extern __shared__ unsigned char sm[];
    // Load the magnitude of the pixels that fall into the block from GM to SM.
    SM_data_loader(sm, sobel_module, sizeSM, radius, rows, cols);

    // Synchronize all threads in the block to make sure all threads have finished writing within the SM.
    __syncthreads();

    // Compute the global indexes of the thread.
    unsigned int y = blockIdx.y*blockDim.y+threadIdx.y;
    unsigned int x = blockIdx.x*blockDim.x+threadIdx.x;

    // Take from the GM la the orientation of the gradient of the pixel under consideration.
    float currDir = sobel_dir[y*cols+x];
    ////
    // I remember that you have to add a + radius shift to the x and y coordinates when accessing the SM ...
    // to make sure that the thread index inside the block matches the pixel index inside the SM matrix.
    ////
    unsigned char mag = sm[(threadIdx.y+radius)*sizeSM+(threadIdx.x+radius)];
    // Normalize the orientation of the gradient.
    while(currDir<0) currDir+=180;

    bool check=true;

    if(y>=rows-1 || y<=0 || x>=cols-1 || x<=0) check=false;
    else{
        // The edges are always located in a direction orthogonal to that of the gradient;
        // Reason for which, we consider the direction orthogonal to that of the gradient in the desired pixel ...
        // and if in this orientation the magnitude of the pixel is greater than that of the two adjacent pixels,
        //then it is considered as an edge pixel.

        if(currDir>22.5 && currDir<=67.5){
            if(mag<sm[(threadIdx.y-1 +radius)*sizeSM+(threadIdx.x-1 +radius)] ||
               mag<sm[(threadIdx.y+1 +radius)*sizeSM+(threadIdx.x+1 +radius)]) check = false;
        }

        else if(currDir>67.5 && currDir<=112.5){
            if(mag<sm[(threadIdx.y-1 +radius)*sizeSM+(threadIdx.x +radius)] ||
               mag<sm[(threadIdx.y+1 +radius)*sizeSM+(threadIdx.x +radius)]) check = false;

        }

        else if(currDir>112.5 && currDir<=157.5){
            if(mag<sm[(threadIdx.y+1 +radius)*sizeSM+(threadIdx.x-1 +radius)] ||
              mag<sm[(threadIdx.y-1 +radius)*sizeSM+(threadIdx.x+1 +radius)]) check = false;

        }

        else{
            if(mag<sm[(threadIdx.y +radius)*sizeSM+(threadIdx.x-1 +radius)] ||
              mag<sm[(threadIdx.y +radius)*sizeSM+(threadIdx.x+1 +radius)]) check = false;
        }

    }
    if(check) out[y*cols+x]=255;
    else out[y*cols+x]=0;

}


__global__ void sobel_kernel(unsigned char* img,
                             unsigned char* sobel_module,
                             float* sobel_dir,
                             unsigned char kernel_size,
                             unsigned char radius,
                             unsigned int sizeSM,
                             short rows,
                             short cols,
                             unsigned char L2_norm){

    // Define shared memory data
    extern __shared__ unsigned char sm[];

    // Upload the necessary data from GM to SM.
    SM_data_loader(sm, img, sizeSM, radius, rows, cols);

    // Synchronize all threads in the block to make sure all threads have finished writing within the SM.
    __syncthreads();

    // Conv step

    // Calculate the derivative with respect to x and y.
    float sumX=0, sumY=0;
    for (int y=0; y<kernel_size; y++)
        for (int x=0; x<kernel_size; x++){
          sumY += sm[(threadIdx.y+y)*sizeSM+(threadIdx.x+x)]*const_weights.y[y][x];
          sumX += sm[(threadIdx.y+y)*sizeSM+(threadIdx.x+x)]*const_weights.x[y][x];
        }

    // Compute the global indexes of the thread.
    unsigned int y = blockIdx.y*blockDim.y+threadIdx.y;
    unsigned int x = blockIdx.x*blockDim.y+threadIdx.x;

    // This check is used to verify that the thread is "contained" within the image.
    if (y<rows && x<cols){

      // Calculate the magnitude of the gredient by applying the norm2 or an approximation of it using the abs.
      int pixel_intensity;
      if(L2_norm==0){
          pixel_intensity = abs(sumY) + abs(sumX);
      }
      else{
          pixel_intensity = sqrt((sumY*sumY)+(sumX*sumX));
      }
      pixel_intensity = pixel_intensity > 255? 255: pixel_intensity < 0? 0: pixel_intensity;

      // Access the GM and save the pixel magnitude value.
      sobel_module[y*cols+x] = pixel_intensity;

      // Calculate the orientation of the gradient, such as arctang (dy / dx).
      sobel_dir[y*cols+x] = atan2(sumY,sumX)*(180/M_PI);

    }
}


void init_kernel_weights(Kernel_weights &k, unsigned char kernel_size){

    memset(&k, 0, sizeof(k));

    if(kernel_size==3){
        k.y[0][0]=1; k.y[0][1]=2; k.y[0][2]=1;
        k.y[1][0]=0; k.y[1][1]=0; k.y[1][2]=0;
        k.y[2][0]=-1; k.y[2][1]=-2; k.y[2][2]=-1;
        //
        k.x[0][0]=1; k.x[0][1]=0; k.x[0][2]=-1;
        k.x[1][0]=2; k.x[1][1]=0; k.x[1][2]=-2;
        k.x[2][0]=1; k.x[2][1]=0; k.x[2][2]=-1;
    }
    else if(kernel_size==5){

    }
    else if(kernel_size==7){

    }


}
