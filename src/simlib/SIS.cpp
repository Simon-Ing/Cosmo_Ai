/* (C) 2023: Hans Georg Schaathun <georg@schaathun.net> */

#include "cosmosim/Lens.h"
#include "simaux.h"

double SIS::psifunction( double x, double y ) {
   return einsteinR*sqrt( x*x + y*y ) ;
}
double SIS::psiXfunction( double x, double y ) {
   double s = sqrt( x*x + y*y ) ;
   return einsteinR*x/s ;
}
double SIS::psiYfunction( double x, double y ) {
   double s = sqrt( x*x + y*y ) ;
   return einsteinR*y/s ;
}

double SIS::getXiAbs( double e ) {
   return (e + einsteinR) ;
}
cv::Point2d SIS::getXi( cv::Point2d chieta ) {
   double phi = atan2(chieta.y, chieta.x); 
   return chieta + einsteinR*cv::Point2d( cos(phi), sin(phi) ) ;
}
