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


std::array<std::array<double, 202>, 201> SIS::getAlphas( cv::Point xi ) {
   throw NotImplemented() ;
}
std::array<std::array<double, 202>, 201> SIS::getBetas( cv::Point xi ) {
   throw NotImplemented() ;
}
void SIS::initAlphasBetas( cv::Point xi ) {
   throw NotImplemented() ;
}
