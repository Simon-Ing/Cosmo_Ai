/* (C) 2023: Hans Georg Schaathun <georg@schaathun.net> */

#include "cosmosim/Lens.h"
#include "simaux.h"

#include <symengine/parser.h>
#include <fstream>

void RouletteLens::updatePsi( ) { 
   throw NotSupported() ;
}
void RouletteLens::updatePsi( cv::Size size ) { 
   throw NotSupported() ;
}
void RouletteLens::setEinsteinR( double r ) { 
   std::cout << "[RouletteLens::setEinsteinR] ignoring.\n" ;
   // throw NotSupported() ;
}

void RouletteLens::calculateAlphaBeta( cv::Point2d xi ) {
   std::cout << "RouletteLens::calculateAlphaBeta() does nothing.\n" ;
}

void RouletteLens::initAlphasBetas() { }

void RouletteLens::setAlphaXi( int m, int s, double val ) {
   alphas_val[m][s] = 0 ;
}
void RouletteLens::setBetaXi( int m, int s, double val ) {
   betas_val[m][s] = 0 ;
}
