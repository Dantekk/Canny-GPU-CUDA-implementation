#include <iostream>
#include <opencv2/opencv.hpp>
#include <cuda.h>
#include "utility.hpp"
#include "CannyGPU.cuh"
#include "GaussianFilter.cuh"
#include "CudaTimer.cuh"

using namespace std;
using namespace cv;
using namespace cuda;

int main(int argc, char **argv){
    /*
    argv[1] -> input image
    argv[2] -> sobel kernel size
    argv[3] -> low threshold
    argv[4] -> high threshold
    argv[5] -> L2 norm -> 0 activated 1 deactivated (uses approximation with abs)
    argv[6] -> modes -> [0] CPU , [1] GPU custom (my implementation) , [2] Runs all modes
    */

    Mat img = imread(argv[1], 0);
    // Declaration of the various Mat structures to contain the intermediate/final steps of the algorithm.
    Mat img_canny_CPU = Mat(img.rows, img.cols, CV_8U, Scalar(0));
    Mat img_filtered = Mat(img.rows, img.cols, CV_8U, Scalar(0));
    Mat img_canny_GPU_custom = Mat(img.rows, img.cols, CV_8U, Scalar(0));
    Mat img_canny_GPU_opencv = Mat(img.rows, img.cols, CV_8U, Scalar(0));
    Mat img_diff = Mat(img.rows, img.cols, CV_8U, Scalar(0));

    // Declaration of the various parameters necessary for the functioning of the algorithm.
    unsigned char kernel_size;
    int low_tr;
    int high_tr;
    int mode_on;
    unsigned char L2_norm;
    double sigma=1.4;

    /// Simple controls on parameters passed from the command line
    if(img.empty()){
        cerr<<"Caricamento dell'immagine fallito!"<<endl;
        exit(1);
    }
    if(argc!=7){
        cerr<<"Numero di parametri non corretto!"<<endl;
        exit(1);
    }
    ///

    kernel_size = (unsigned char)atoi(argv[2]);
    low_tr = atoi(argv[3]);
    high_tr = atoi(argv[4]);
    L2_norm = (unsigned char)atoi(argv[5]);
    mode_on = atoi(argv[6]);

    int rows = img.rows; //y
    int cols = img.cols; //x

    // Create a CudiTimer object -> it's the class I created to manage timing.
    CudaTimer cuda_timer;

    // Image Smoothing
    GaussianBlur(img, img_filtered, Size(3,3), sigma);
    //img_filtered = img;
    //GaussianFilterGPU(img_host, img_out_host_gaussian, 3, 1.4, rows, cols);

    if(mode_on==0 || mode_on==2){
        // Canny CPU

        ///
        cuda_timer.start_timer();
        CannyCPU(img_filtered, img_canny_CPU, kernel_size, L2_norm, low_tr, high_tr);
        cuda_timer.stop_timer();
        ///
        printf("Tempo esecuzione Canny CPU : %f ms\n", cuda_timer.get_time());
        imwrite("Canny_imp_CPU.jpg", img_canny_CPU);
    }
    if(mode_on==1 || mode_on==2){
        // Canny GPU
        unsigned char *img_host = (unsigned char*)malloc(rows*cols*sizeof(unsigned char));
        // Converts the image passing from a Mat Opencv type structure
        // to a dynamically allocated 2D matrix.
        convertImg(img_filtered, img_host, rows, cols);

        unsigned char *img_out_host = (unsigned char*)malloc(rows*cols*sizeof(unsigned char));
        unsigned char *img_out_host_gaussian = (unsigned char*)malloc(rows*cols*sizeof(unsigned char));

        ////
        cuda_timer.start_timer();
        CannyGPU(img_host, img_out_host, rows, cols, kernel_size, low_tr, high_tr, L2_norm);
        cuda_timer.stop_timer();
        printf("Tempo esecuzione Canny GPU custom : %f ms\n", cuda_timer.get_time());
        ////

        // Convert the image by passing to a structure of type Mat Opencv.
        // This facilitates saving the image to disk using the OpenCV functions.
        convertImg2(img_canny_GPU_custom, img_out_host, rows, cols);
        imwrite("Canny_imp_GPU_custom.jpg", img_canny_GPU_custom);

        free(img_host);
        free(img_out_host);
    }

    exit(1);

}
