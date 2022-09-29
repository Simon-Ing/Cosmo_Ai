/* (C) 2022: Hans Georg Schaathun <hg@schaathun.net> */

#include "Simulator.h"
#include <thread>

EllipsoidSource::EllipsoidSource( int sz, double sig1, double sig2 ) :
        size(sz),
        sigma1(sig1),
        sigma2(sig2)
{ 
    // imgActual = cv::Mat(size, size, CV_8UC1, cv::Scalar(0, 0, 0));
    imgApparent = cv::Mat(size, size, CV_8UC1, cv::Scalar(0, 0, 0));
    drawParallel( imgApparent ) ;
}



/* Draw the source image.  The sourceSize is interpreted as the standard deviation in a Gaussian distribution */
void Source::drawSource(int begin, int end, cv::Mat& dst) {
    for (int row = begin; row < end; row++) {
        for (int col = 0; col < dst.cols; col++) {
            int x = col - dst.cols/2;
            int y = row - dst.rows/2;
            auto value = (uchar)round(255 * exp( 0.5*(-(x*x)/(sigma1*sigma1) - (y*y)/(sigma2*sigma2) ) ));
            dst.at<uchar>(row, col) = value;
        }
    }
}

