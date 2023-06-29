/* (C) 2023: Hans Georg Schaathun <georg@schaathun.net> */

#include "cosmosim/Roulette.h"
#include "simaux.h"

void RouletteRegenerator::updateApparentAbs( ) {
    std::cout << "[RouletteRegenerator] updateApparentAbs() does nothing.\n" ;
}

void RouletteRegenerator::setCentre( cv::Point2d pt ) {
   setNu( cv::Point2d( 0,0 ) ) ;
   setXY( -pt.x, -pt.y, CHI, einsteinR ) ;
   etaOffset = pt ;
   std::cout << "[LensModel::setCentre] etaOffset = " << etaOffset 
        << "; nu=" << getNu() << "; eta=" << getEta() << "; xi=" << xi << "\n" ;
}
