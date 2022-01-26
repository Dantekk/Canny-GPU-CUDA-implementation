#include <opencv2/opencv.hpp>

using namespace cv;
void convertImg(Mat img, unsigned char* out_img, int rows, int cols);
void convertImg2(Mat &img, unsigned char* out_img, int rows, int cols);
void CannyCPU(Mat src, Mat &dest, int kernel_size, int L2norm, int low_tr, int high_tr);
