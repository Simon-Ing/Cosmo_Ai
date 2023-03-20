/* (C) 2023: Hans Georg Schaathun <georg@schaathun.net> */

#include "cosmosim/Simulator.h"

// Add some lines to the image for reference
void refLines(cv::Mat& target) {
    int rsize = target.rows;
    int csize = target.cols;
    std::cout << "refLines " << rsize << "x" << csize << "\n" ;
    cv::line( target, cv::Point( 0, csize /2 ),
                      cv::Point( rsize, csize /2 ),
                      {60, 60, 60},
                      1 ) ;
    cv::line( target, cv::Point( rsize/2-1, 0 ),
                      cv::Point( rsize/2-1, csize ),
                      {60, 60, 60},
                      1 ) ;
    cv::line( target, cv::Point( 0, csize-1 ),
                      cv::Point( rsize, csize-1 ),
                      {255, 255, 255},
                      1 ) ;
    cv::line( target, cv::Point( rsize-1, 0 ),
                      cv::Point( rsize-1, csize ),
                      {255, 255, 255},
                      1 ) ;
    cv::line( target, cv::Point( 0, 0 ),
                      cv::Point( rsize, 0 ),
                      {255, 255, 255},
                      1 ) ;
    cv::line( target, cv::Point( 0, 0 ),
                      cv::Point( 0, csize ),
                      {255, 255, 255},
                      1 ) ;
}

/* Calculate n! (n factorial) */
double factorial_(unsigned int n){
    double a = 1.0;
    for (int i = 2; i <= n; i++){
        a *= i;
    }
    return a;
}

/* Notes on the Sobel filter below.
 * - Sobel() does not normalise the filter by default, hence 
 *   the scaling factor of 1.0/8.
 * - The X direction is vertical and Y is horizontal.
 * - Convolution flips the filter and hence the sign; therefore
 *   we use a negative scaling factor in the horizontal filter.
 * - Vertical indexing increasing from top to bottom, and
 *   hence we flip the sign a second time to get a positive scale 
 *   in this direction.
 */

void gradient(cv::InputArray src, cv::OutputArray outX, cv::OutputArray outY) {
   Sobel(src, outX, CV_64F, 1, 0, 3, 1.0/8 ) ;
   Sobel(src, outY, CV_64F, 0, 1, 3, -1.0/8 ) ;
   // Sobel(src, out, ddepth, 1, 0, ksize, scale, delta, BORDER_DEFAULT);
   // Sobel(src, out, ddepth, 0, 1, ksize, scale, delta, BORDER_DEFAULT);
}

cv::Point2d imageCoordinate( cv::Point2d pt, cv::Mat im ) {
   int ncols=im.cols, nrows=im.rows ;
   return cv::Point2d( nrows/2 - pt.y, pt.x + ncols/2 ) ;
}
cv::Point2d pointCoordinate( cv::Point2d pt, cv::Mat im ) {
   int ncols=im.cols, nrows=im.rows ;
   return cv::Point2d( pt.y - ncols/2, nrows/2 - pt.x ) ;
}
