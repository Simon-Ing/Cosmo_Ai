/* (C) 2023: Hans Georg Schaathun <georg@schaathun.net> */

#include "cosmosim/Roulette.h"
#include "simaux.h"

void RouletteRegenerator::updateApparentAbs( ) {
    std::cout << "[RouletteRegenerator] updateApparentAbs() does nothing.\n" ;
}

void RouletteRegenerator::setCentre( cv::Point2d pt, cv::Point2d eta ) {
   setNu( cv::Point2d( 0,0 ) ) ;
   setXY( eta.x, eta.y, CHI, einsteinR ) ;
   etaOffset = pt ;
   std::cout << "[LensModel::setCentre] etaOffset = " << etaOffset 
        << "; nu=" << getNu() << "; eta=" << getEta() << "; xi=" << xi << "\n" ;
}
void RouletteRegenerator::setAlphaXi( int m, int s, double val ) {
   alphas_val[m][s] = val ;
}
void RouletteRegenerator::setBetaXi( int m, int s, double val ) {
   betas_val[m][s] = val ;
}
void RouletteRegenerator::calculateAlphaBeta() { 
   std::cout << "[RouletteRegenerator] calculateAlphaBeta() does nothuing."
      << std::endl << std::flush ;
} 
