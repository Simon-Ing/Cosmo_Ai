/* (C) 2022: Hans Georg Schaathun <hg@schaathun.net> */

#include "Simulator.h"
#include <thread>

TriangleSource::TriangleSource( int sz, double sig, double thet ) :
        sigma(sig),
        theta(thet),
        Source::Source(sz)
{ }
TriangleSource::TriangleSource( int sz, double sig ) :
        TriangleSource(sz,sig,0)
{ }


void TriangleSource::drawSource(int begin, int end, cv::Mat& dst) {
   std::cout << "TriangleSource::drawSource() - not implemented\n" ;
}

/* drawParallel() split the image into chunks to draw it in parallel using drawSource() */
void TriangleSource::drawParallel(cv::Mat& dst){
   std::cout << "TriangleSource::drawParallel() \n" ;
    int r0 = dst.rows/2, c0 = dst.cols/2;
    cv::Point pt1 = cv::Point( r0+sigma, c0 ) ;
    cv::Point pt2 = cv::Point( r0+cos(2*PI/3)*sigma, c0+sin(2*PI/3)*sigma ) ;
    cv::Point pt3 = cv::Point( r0+cos(2*PI/3)*sigma, c0-sin(2*PI/3)*sigma ) ;

    std::cout << dst.type() << pt1 << pt2 << pt3 << "\n" ;

    cv::line( dst, pt1, pt2, {192, 0, 0}, 3 ) ;
    cv::line( dst, pt2, pt3, {0, 192, 0}, 3 ) ;
    cv::line( dst, pt3, pt1, {0, 0, 192}, 3 ) ;

    cv::Mat rot = cv::getRotationMatrix2D( cv::Point(dst.rows/2, dst.cols/2), theta*180/PI, 1);
    cv::warpAffine(dst, dst, rot, dst.size());    // crop distorted image
}

