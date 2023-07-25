/* (C) 2023: Hans Georg Schaathun <georg@schaathun.net> */

#include "cosmosim/Lens.h"
#include "simaux.h"

// \xi_0 ~ einsteinR
// NEW f
// NEW \phi
// TODO \chi_L in terms of \chi

double SIE::psifunction( double x, double y ) {
   double sq = sqrt( 1 - ellipseratio*ellipseratio ) ;
   double sqf = sqrt( ellipseratio )/sq ;
   double theta = phi + atan2(y, x); 
   double sintheta = sin(theta), costheta = cos(theta) ;
   double R = sqrt( x*x + y*y ) ;
   return einsteinR*R*sqf*(
	   sintheta*asin( sq * sintheta )
	   + costheta*asinh(costheta*sq/ellipseratio)
	 ) ;
}
double SIE::psiXfunction( double x, double y ) {
   throw NotImplemented() ;
   double s = sqrt( x*x + y*y ) ;
   return einsteinR*x/s ;
}
double SIE::psiYfunction( double x, double y ) {
   throw NotImplemented() ;
   double s = sqrt( x*x + y*y ) ;
   return einsteinR*y/s ;
}

cv::Point2d SIE::getXi( cv::Point2d chieta ) {
   throw NotImplemented() ;
   double phi = atan2(chieta.y, chieta.x); 
   return chieta + einsteinR*cv::Point2d( cos(phi), sin(phi) ) ;
}
