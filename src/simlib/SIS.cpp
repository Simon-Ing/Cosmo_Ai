/* (C) 2023: Hans Georg Schaathun <georg@schaathun.net> */

#include "cosmosim/Lens.h"
#include "simaux.h"

#define norm(x,y) sqrt( x*x + y*y ) 

std::string SIS::idString() {
   return "SIS" ;
};


cv::Point2d SIS::getXi( cv::Point2d eta ) {
   cv::Point2d r = eta + cv::Point2d( 
         psiXvalue(eta.x, eta.y ),
         psiYvalue(eta.x, eta.y ) ) ;
   if (DEBUG) std::cout << "[SIS::getXi] " << eta << " -> " << r << "\n" ;
   return r ;
}

double SIS::psiValue( double x, double y ) const {
   return einsteinR*sqrt( x*x + y*y ) ;
}
double SIS::psiXvalue( double x, double y ) const {
   double s = sqrt( x*x + y*y ) ;
   return einsteinR*x/s ;
}
double SIS::psiYvalue( double x, double y ) const {
   double s = sqrt( x*x + y*y ) ;
   return einsteinR*y/s ;
}

double SIS::criticalXi( double phi ) const {
   return einsteinR ;
}
cv::Point2d SIS::caustic( double phi ) const {

   return cv::Point2d( 0, 0 ) ;
   // return einsteinR*cv::Point2d(  cos(phi), sin(phi) ) ;
}
