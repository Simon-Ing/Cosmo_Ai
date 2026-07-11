/* (C) 2023: Hans Georg Schaathun <georg@schaathun.net> */

#include "cosmosim/Simulator.h"

#define signf(y)  ( y < 0 ? -1 : +1 )

// Add some lines to the image for reference
// void refLines(cv::Mat& ) ;
double factorial_(unsigned int) ;
void diffX(cv::InputArray, cv::OutputArray ) ;
void diffY(cv::InputArray, cv::OutputArray ) ;

cv::Point2d pointCoordinate( cv::Point2d pt, cv::Mat im ) ;
cv::Point2d imageCoordinate( cv::Point2d pt, cv::Mat im ) ;
void gradient(cv::InputArray, cv::OutputArray, cv::OutputArray ) ;

