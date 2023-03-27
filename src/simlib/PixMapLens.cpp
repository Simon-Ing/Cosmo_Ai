/* (C) 2023: Hans Georg Schaathun <georg@schaathun.net> */

#include "cosmosim/PixMap.h"
#include "cosmosim/Lens.h"
#include "simaux.h"

void PixMapLens::updatePsi( cv::Size size ) { 
   return ; 
}

void LensMap::setPsi( cv::Mat map ) {
   cv::Mat tmp ;
   std::cout << "[LensMap] setPsi()\n" ;
   psi = map ;
   // diffX( psi, tmp ) ; diffX( tmp, psiX ) ;
   // diffY( psi, tmp ) ; diffY( tmp, psiY ) ;
   Sobel(psi,psiX,CV_64FC1, 2, 0, 3, 1.0/8) ;
   Sobel(psi,psiY,CV_64FC1, 0, 2, 3, 1.0/8) ;
   massMap = ( psiX + psiY ) / 2 ;

   // Calculate einsteinMap 
}
void LensMap::loadPsi( std::string fn ) {
   setPsi( cv::imread( fn ) ) ;
   // Calculate einsteinMap and massMap here
}
