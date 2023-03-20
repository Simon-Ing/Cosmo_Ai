/* (C) 2022: Hans Georg Schaathun <georg@schaathun.net> */

#include "cosmosim/Roulette.h"

double factorial_(unsigned int n);

cv::Point getOrigin( cv::Point R, double phi, double x0, double y0 ) {
      double c = cos(phi), s = R.x*sin(phi) ;
      double x = x0 + R.x*c - R.y*s,
             y = y0 - (R.x*s + R.y*c) ;
      return cv::Point( x, y ) ;
}
void RouletteLens::maskImage( cv::InputOutputArray imgD ) {
      std::cout << "RouletteLens::maskImage\n" ;
      cv::Point origo = getOrigin( getCentre(), phi, imgD.cols()/2, imgD.rows()/2 ) ;
      cv::Mat mask( imgD.size(), CV_8UC1, cv::Scalar(255) ) ;
      cv::Mat black( imgD.size(), imgD.type(), cv::Scalar(0) ) ;
      cv::circle( mask, origo, getMaskRadius(), cv::Scalar(0), cv::FILLED ) ;
      black.copyTo( imgD, mask ) ;
}
void RouletteLens::markMask( cv::InputOutputArray imgD ) {
      std::cout << "RouletteLens::maskImage\n" ;
      cv::Point origo = getOrigin( getCentre(), phi, imgD.cols()/2, imgD.rows()/2 ) ;
      cv::circle( imgD, origo, getMaskRadius(), cv::Scalar(255), 1 ) ;
      cv::circle( imgD, origo, 3, cv::Scalar(0), 1 ) ;
      cv::circle( imgD, origo, 1, cv::Scalar(0), cv::FILLED ) ;
}

double RouletteLens::getMaskRadius() const { return getNuAbs() ; }
