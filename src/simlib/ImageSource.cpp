/* (C) 2022: Hans Georg Schaathun <georg@schaathun.net> */

/* The image source takes an arbitrary image to use as the source.
 * This is mainly for show, sporting an photo of Einstein in the tool
 * implementations.
 */

#include "cosmosim/Source.h"

#include <thread>

ImageSource::ImageSource( int sz,  std::string fn ) :
   Source(sz) {
   filename = fn ;
}

/* Getters for the images */
cv::Mat ImageSource::getImage() { 
   if ( ! drawn ) {
      std::cout << "ImageSource.getImage() loading image " << size << "\n" ;
      cv::Mat img = cv::imread( filename );
      imgApparent = cv::Mat( size, size, img.type(), cv::Scalar(0,0,0));
      imgApparent.setTo(cv::Scalar::all(0));

      int x = (size - img.rows)/2,
          y = (size - img.cols)/2 ;
      img.copyTo(imgApparent(cv::Rect(x, y, img.cols, img.rows)));

      drawn = 1 ;
   }


   std::cout << "ImageSource.getImage() returns\n" ;
   return imgApparent ; 
}
void ImageSource::drawSource( int x, int y, cv::Mat& dst ) {
   std::cout << "ImageSource::drawSource() does nothing.\n" ;
}
