/* (C) 2022: Hans Georg Schaathun <hg@schaathun.net> */

#include "Source.h"
#include <thread>

EllipsoidSource::EllipsoidSource( int sz, double sig1, double sig2 ) :
        Source::Source(sz),
        sigma1(sig1),
        sigma2(sig2)
{ }


/* Draw the source image.  The sourceSize is interpreted as the standard deviation in a Gaussian distribution */
void EllipsoidSource::drawSource(int begin, int end, cv::Mat& dst) {
    for (int row = begin; row < end; row++) {
        for (int col = 0; col < dst.cols; col++) {
            int x = col - dst.cols/2;
            int y = row - dst.rows/2;
            auto value = (uchar)round(255 * exp( 0.5*(-(x*x)/(sigma1*sigma1) - (y*y)/(sigma2*sigma2) ) ));
            dst.at<uchar>(row, col) = value;
        }
    }
}

