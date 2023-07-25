/* (C) 2023: Hans Georg Schaathun <georg@schaathun.net> */

#include "cosmosim/Lens.h"
#include "simaux.h"

double PointMass::psifunction( double x, double y ) {
   return einsteinR*sqrt( x*x + y*y ) ;
}
double PointMass::psiXfunction( double x, double y ) {
   double s = sqrt( x*x + y*y ) ;
   return einsteinR*x/s ;
}
double PointMass::psiYfunction( double x, double y ) {
   double s = sqrt( x*x + y*y ) ;
   return einsteinR*y/s ;
}

cv::Point2d PointMass::getXi( cv::Point2d chieta ) {
   double c = chieta.x*chieta.x + chieta.y*chieta.y ;
   return cv::Point2d( sqrt(c)/2 + sqrt(0.25*c + einsteinR*einsteinR), 0 ) ;
}
