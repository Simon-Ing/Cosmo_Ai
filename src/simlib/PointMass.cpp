/* (C) 2023: Hans Georg Schaathun <georg@schaathun.net> */

/* Note.  This is a hack to make PointMassLens and RoulettePMLens work
 * and share the getXI() calculation.  */

#include "cosmosim/Lens.h"
#include "simaux.h"

double PointMass::psifunction( double x, double y ) {
   double r = sqrt( x*x + y*y ) ;
   return (einsteinR*einsteinR)*log(r/einsteinR) ;
}
double PointMass::psiXfunction( double x, double y ) {
   double r = sqrt( x*x + y*y ) ;
   return (einsteinR*einsteinR)*(einsteinR/r)*(x/r) ;
}
double PointMass::psiYfunction( double x, double y ) {
   double r = sqrt( x*x + y*y ) ;
   return (einsteinR*einsteinR)*(einsteinR/r)*(y/r) ;
}

cv::Point2d PointMass::getXi( cv::Point2d chieta ) {
   double phi = atan2(chieta.y, chieta.x); 
   double c = chieta.x*chieta.x + chieta.y*chieta.y ;
   double a = sqrt(c)/2 + sqrt(0.25*c + einsteinR*einsteinR) ;
   return chieta + a*cv::Point2d( cos(phi), sin(phi) ) ;
}
