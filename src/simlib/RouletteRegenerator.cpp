/* (C) 2023: Hans Georg Schaathun <georg@schaathun.net> */

#include "cosmosim/Roulette.h"
#include "simaux.h"

void RouletteRegenerator::updateApparentAbs( ) {
    std::cout << "[RouletteRegenerator] updateApparentAbs() does nothing.\n" ;
}

void RouletteRegenerator::setCentrePy( double x, double y ) {
   this->setCentre( cv::Point2d(x,y), cv::Point2d( 0,0 ) ) ;
}

void RouletteRegenerator::setCentre( cv::Point2d pt, cv::Point2d eta ) {
   setNu( cv::Point2d( 0,0 ) ) ;
   setXY( eta.x, eta.y ) ;
   etaOffset = pt ;
   if (DEBUG) std::cout << "[RouletteRegenerator::setCentre] etaOffset = " 
        << etaOffset 
        << "; nu=" << getNu() << "; eta=" << getEta() << "; xi=" << getXi() << "\n" ;
}
void RouletteRegenerator::setAlphaXi( int m, int s, double val ) {
   alphas_val[m][s] = val ;
}
void RouletteRegenerator::setBetaXi( int m, int s, double val ) {
   betas_val[m][s] = val ;
}
