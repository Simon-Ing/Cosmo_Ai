/* (C) 2023: Hans Georg Schaathun <georg@schaathun.net> */

#include "cosmosim/Lens.h"
#include "simaux.h"

#include <symengine/parser.h>
#include <fstream>

void RouletteLens::updatePsi( ) { 
   std::cout << "RouletteLens::updatePsi()\n" ;
   // throw NotSupported() ;
}
void RouletteLens::updatePsi( cv::Size size ) { 
   updatePsi() ;
}
void RouletteLens::setEinsteinR( double r ) { 
   std::cout << "[RouletteLens::setEinsteinR] ignoring.\n" ;
   // throw NotSupported() ;
}

void RouletteLens::calculateAlphaBeta( cv::Point2d xi ) {
   std::cout << "RouletteLens::calculateAlphaBeta() does nothing.\n" ;
   // We should have checked that xi=0 to avoid accidental misuse.
}

void RouletteLens::initAlphasBetas() { }

void RouletteLens::setAlphaXi( int m, int s, double val ) {
   alphas_val[m][s] = val ;
}
void RouletteLens::setBetaXi( int m, int s, double val ) {
   betas_val[m][s] = val ;
}
cv::Point2d RouletteLens::getXi( cv::Point2d eta ) {
   return cv::Point2d( 0, 0 ) ;
}
