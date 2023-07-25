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
   double theta = phi + atan2(y, x); 
   double sint = sin(theta) ;
   double cost = cos(theta) ;
   double f = ellipseratio ;
   double f2 = f*f, cost2 = cost*cost, sint2 = sint*sint ;
   return -sqrt(f)*einsteinR*(x*sqrt(f2*sint2 - sint2 + 1)*sqrt(-f2*cost2 + f2 + cost2)*sint*asin(sqrt(1 - f2)*sint) + x*sqrt(f2*sint2 - sint2 + 1)*sqrt(-f2*cost2 + f2 + cost2)*cost*asinh(sqrt(1 - f2)*cost/f) + y*sqrt(1 - f2)*sqrt(f2*sint2 - sint2 + 1)*sint*cost - y*sqrt(1 - f2)*sqrt(-f2*cost2 + f2 + cost2)*sint*cost + y*sqrt(f2*sint2 - sint2 + 1)*sqrt(-f2*cost2 + f2 + cost2)*sint*asinh(sqrt(1 - f2)*cost/f) - y*sqrt(f2*sint2 - sint2 + 1)*sqrt(-f2*cost2 + f2 + cost2)*cost*asin(sqrt(1 - f2)*sint)
	 )*sqrt(1/(1 - f2))
      / (sqrt(x*x + y*y)*sqrt(f2*sint2 - sint2 + 1)*sqrt(-f2*cost2 + f2 + cost2)) ;
}
double SIE::psiYfunction( double x, double y ) {
   double theta = phi + atan2(y, x); 
   double sint = sin(theta) ;
   double cost = cos(theta) ;
   double f = ellipseratio ;
   double f2 = f*f, cost2 = cost*cost, sint2 = sint*sint ;
   return
      -sqrt(f)*einsteinR*(-x*sqrt(1 - f2)*sqrt(f2*sint2 - sint2 + 1)*sint*cost + x*sqrt(1 - f2)*sqrt(-f2*cost2 + f2 + cost2)*sint*cost - x*sqrt(f2*sint2 - sint2 + 1)*sqrt(-f2*cost2 + f2 + cost2)*sint*asinh(sqrt(1 - f2)*cost/f) + x*sqrt(f2*sint2 - sint2 + 1)*sqrt(-f2*cost2 + f2 + cost2)*cost*asin(sqrt(1 - f2)*sint) + y*sqrt(f2*sint2 - sint2 + 1)*sqrt(-f2*cost2 + f2 + cost2)*sint*asin(sqrt(1 - f2)*sint) + y*sqrt(f2*sint2 - sint2 + 1)*sqrt(-f2*cost2 + f2 + cost2)*cost*asinh(sqrt(1 - f2)*cost/f))*sqrt(1/(1 - f2))
      / (sqrt(x*x + y*y)*sqrt(f2*sint2 - sint2 + 1)*sqrt(-f2*cost2 + f2 + cost2)) ;
}

cv::Point2d SIE::getXi( cv::Point2d chieta ) {
   throw NotImplemented() ;
   double phi = atan2(chieta.y, chieta.x); 
   return chieta + einsteinR*cv::Point2d( cos(phi), sin(phi) ) ;
}
