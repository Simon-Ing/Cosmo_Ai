/* (C) 2023: Hans Georg Schaathun <georg@schaathun.net> */

#include "cosmosim/PixMap.h"

#include "simaux.h"

cv::Mat LensMap::getPsi() const {
   return psi ;
}
cv::Mat LensMap::getPsiImage() const {
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
cv::Mat LensMap::getMassMap() const {
   return massMap ;
}
cv::Mat LensMap::getMassImage() const {
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
cv::Mat LensMap::getEinsteinMap() const {
   throw NotImplemented() ;
   // return einsteinMap ;
}

void LensMap::setPsi( cv::Mat map ) {
   cv::Mat tmp, psix, psiy ;
   std::cout << "[LensMap] setPsi()\n" ;
   psi = map ;
   // diffX( psi, tmp ) ; diffX( tmp, psix ) ;
   // diffY( psi, tmp ) ; diffY( tmp, psiy ) ;
   Sobel(psi,psix,CV_64FC1, 2, 0, 3, 1.0/8) ;
   Sobel(psi,psiy,CV_64FC1, 0, 2, 3, 1.0/8) ;
   massMap = ( psix + psiy ) / 2 ;

   // Calculate einsteinMap 
}
void LensMap::loadPsi( std::string fn ) {
   setPsi( cv::imread( fn ) ) ;
   // Calculate einsteinMap and massMap here
}
void LensMap::updatePsi() { 
   return ;
}
