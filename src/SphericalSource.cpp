/* (C) 2022: Hans Georg Schaathun <hg@schaathun.net> */

/* The SphericalSource class implements a a Spherical, Gaussian mass, */

#include "Source.h"
#include <thread>

SphericalSource::SphericalSource(int sz,double sig) :
        Source::Source(sz),
        sigma(sig)
{ }

/* Draw the source image.  The sourceSize is interpreted as the standard deviation in a Gaussian distribution */
void SphericalSource::drawSource(int begin, int end, cv::Mat& dst) {
    for (int row = begin; row < end; row++) {
        for (int col = 0; col < dst.cols; col++) {
            int x = col - dst.cols/2;
            int y = row - dst.rows/2;
            auto value = (uchar)round(255 * exp((-x * x - y * y) / (2.0*sigma*sigma)));
            dst.at<uchar>(row, col) = value;
        }
    }
}

