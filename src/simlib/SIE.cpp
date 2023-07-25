/* (C) 2023: Hans Georg Schaathun <georg@schaathun.net> */

#include "cosmosim/Lens.h"
#include "simaux.h"

// \xi_0 ~ einsteinR
// NEW f
// NEW \phi
// TODO \chi_L in terms of \chi

double SIE::psifunction( double x, double y ) {
   return einsteinR*sqrt( x*x + y*y ) ;
}
double SIE::psiXfunction( double x, double y ) {
   double s = sqrt( x*x + y*y ) ;
   return einsteinR*x/s ;
}
double SIE::psiYfunction( double x, double y ) {
   double s = sqrt( x*x + y*y ) ;
   return einsteinR*y/s ;
}

double SIE::getXiAbs( double e ) {
   return (e + einsteinR) ;
}
cv::Point2d SIE::getXi( cv::Point2d chieta ) {
   double phi = atan2(chieta.y, chieta.x); 
   return chieta + einsteinR*cv::Point2d( cos(phi), sin(phi) ) ;
}
