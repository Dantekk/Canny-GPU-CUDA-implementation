#include "utility.hpp"

void convertImg2(Mat &img, unsigned char* out_img, int rows, int cols){

    for(int i=0; i<rows; i++)
        for(int j=0; j<cols; j++)
            img.at<uchar>(i,j) = out_img[i*cols+j];

}

void convertImg(Mat img, unsigned char* out_img, int rows, int cols){

    for(int i=0; i<rows; i++)
        for(int j=0; j<cols; j++)
            out_img[i*cols+j] = img.at<uchar>(i,j);

}


void CannyCPU(Mat src, Mat &dest, int kernel_size, int L2norm, int low_tr, int high_tr){

    bool L2gradient = false;
    if(L2norm==1) L2gradient=true;

    Canny(src, dest, low_tr, high_tr, kernel_size, L2gradient);
}
