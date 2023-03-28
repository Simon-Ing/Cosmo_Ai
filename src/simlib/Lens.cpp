/* (C) 2023: Hans Georg Schaathun <georg@schaathun.net> */

#include "cosmosim/Lens.h"
#include "simaux.h"

void Lens::updatePsi( ) { 
   return updatePsi( cv::Size(400,400) ) ;
}
void Lens::updatePsi( cv::Size size ) { 
   return ; 
}
void Lens::setEinsteinR( double r ) { einsteinR = r ; }

cv::Mat Lens::getPsi() const {
   return psi ;
}
cv::Mat Lens::getPsiX() const {
   return psiX ;
}
cv::Mat Lens::getPsiY() const {
   return psiY ;
}
cv::Mat Lens::getPsiImage() const {
   cv::Mat im, ps = getPsi() ;
   double minVal, maxVal;
   cv::Point minLoc, maxLoc;
   minMaxLoc( ps, &minVal, &maxVal, &minLoc, &maxLoc ) ;
   double s = std::max(-minVal,maxVal)*128 ;
   ps /= s ;
   ps += 128 ;
   ps.convertTo( im, CV_8S ) ;
   return im ;
}
cv::Mat Lens::getMassMap() const {
   cv::Mat psiX2, psiY2 ;
   Sobel(psi,psiX,CV_64FC1, 2, 0, 3, 1.0/8) ;
   Sobel(psi,psiY,CV_64FC1, 0, 2, 3, 1.0/8) ;
   return ( psiX + psiY ) / 2 ;
}
cv::Mat Lens::getMassImage() const {
   cv::Mat im, k ;
   double minVal, maxVal;
   cv::Point minLoc, maxLoc;

   k = getMassMap() ;
   minMaxLoc( k, &minVal, &maxVal, &minLoc, &maxLoc ) ;
   assert ( minVal >= 0 ) ;
   if ( maxVal > 255 ) {
     k /= maxVal ;
     k *= 255 ;
   }
   k.convertTo( im, CV_8S ) ;
   return im ;
}
cv::Mat Lens::getEinsteinMap() const {
   throw NotImplemented() ;
   // return einsteinMap ;
}
