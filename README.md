# Canny-GPU
CUDA implementation of Canny edge detector in C/C++. </br>
You can use cmake to compile the files. I have made a CMakeLists available for compilation. </br>
## Run the code
I have made available a main file that executes the code. </br>
In particular, these are the parameters to be given on the command line: </br></br>
`
./main argv[1] argv[2] argv[3] argv[4] argv[5] argv[6]
`
</br></br>
where :
* `argv[1]` : input image path
* `argv[2]` : kernel size of Sobel
* `argv[3]` : low threshold for Hysteresis step
* `argv[4]` : high threshold for Hysteresis step
* `argv[5]` : L2 norm -> 0 activated 1 deactivated (uses approximation with abs)
* `argv[6]` : modes -> [0] CPU , [1] GPU custom (my implementation) , [2] Runs all modes. With [0] run OpenCV Canny CPU while with [1] run Opencv GPU. At last, with [2] run both.

_During the execution of the algorithm, the execution times are also calculated, expressed in ms._ 
## Results example
Examples of image output of my Canny GPU version.
| Original             |  Canny GPU Output | 
:-------------------------:|:-------------------------: | 
![](https://github.com/Dantekk/Canny-GPU-CUDA-implementation/blob/main/images/output_test/oc.png)  |  ![](https://github.com/Dantekk/Canny-GPU-CUDA-implementation/blob/main/images/output_test/oc_imp_GPU_custom.jpg) 

| Original             |  Canny GPU Output | 
:-------------------------:|:-------------------------: | 
![](https://github.com/Dantekk/Canny-GPU-CUDA-implementation/blob/main/images/output_test/circles.png)  |  ![](https://github.com/Dantekk/Canny-GPU-CUDA-implementation/blob/main/images/output_test/circles_custom.jpg) 

| Original             |  Canny GPU Output | 
:-------------------------:|:-------------------------: | 
![](https://github.com/Dantekk/Canny-GPU-CUDA-implementation/blob/main/images/output_test/chessboard.jpg)  |  ![](https://github.com/Dantekk/Canny-GPU-CUDA-implementation/blob/main/images/output_test/chess_custom.jpg) 

**N.B:** obviously, the results may vary according to the value chosen for the thresholds in the hysteresis step.
## Kernel config
I tried several kernel configurations but the one that gave the best results was the one where I used a thread block size of 16x16.

| Kernel Configuration |  
:-------------------------:|
![](https://github.com/Dantekk/Canny-GPU-CUDA-implementation/blob/main/images/test/kernel_config_high.jpg)
## Kernel time esecution
This is the pie chart showing the execution times of the various kernel device function and data transfer memcpy routines on 720p image resolution.
| Kernel time esec |  
:-------------------------:|
![](https://github.com/Dantekk/Canny-GPU-CUDA-implementation/blob/main/images/test/kernel_call_720_high.jpg)
## CPU v.s. GPU
This is the comparison analysis between the OpenCV CPU version and my parallel version on GPU.
| CPU v.s. GPU |  
:-------------------------:|
![](https://github.com/Dantekk/Canny-GPU-CUDA-implementation/blob/main/images/test/cpu_vs_gpu_high2.jpg)

As you can see from the graph, with a low resolution image the results of the two versions are similar. As the image resolution increases, the parallel version gets significantly better results.
