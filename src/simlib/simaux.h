/* (C) 2022: Hans Georg Schaathun <georg@schaathun.net> */

#include "cosmosim/Simulator.h"

// Add some lines to the image for reference
void refLines(cv::Mat& ) ;
double factorial_(unsigned int) ;
void diffX(cv::InputArray, cv::OutputArray ) ;
void diffY(cv::InputArray, cv::OutputArray ) ;

cv::Point2d pointCoordinate( cv::Point2d pt, cv::Mat im ) ;
cv::Point2d imageCoordinate( cv::Point2d pt, cv::Mat im ) ;
void gradient(cv::InputArray, cv::OutputArray, cv::OutputArray ) ;
// #define diffX(src,out) Sobel(src,out,CV_64FC1, 1, 0, 3, 1.0/8)
// #define diffY(src,out) Sobel(src,out,CV_64FC1, 0, 1, 3, -1.0/8)
