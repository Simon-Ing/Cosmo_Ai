/* (C) 2023: Hans Georg Schaathun <georg@schaathun.net> */

#include "cosmosim/Roulette.h"

#include <symengine/expression.h>
#include <symengine/lambda_double.h>
#include <symengine/parser.h>

#include <thread>
#include <fstream>

RouletteSISLens::RouletteSISLens() :
   SphereLens::SphereLens()
{ 
    std::cout << "Instantiating RouletteSISLens ... \n" ;
    rotatedMode = false ;
}
RouletteSISLens::RouletteSISLens(bool centred) :
   SphereLens::SphereLens(centred)
{ 
    std::cout << "Instantiating RouletteSISLens ... \n" ;
    rotatedMode = false ;
}
RouletteSISLens::RouletteSISLens(std::string fn, bool centred) :
   SphereLens::SphereLens(fn,centred)
{ 
    std::cout << "Instantiating RouletteSISLens ... \n" ;
    rotatedMode = false ;
}
void RouletteSISLens::updateApparentAbs( ) {
    double r = getEtaAbs() + einsteinR/CHI ;
    setNu( cv::Point2d( r*cos(phi), r*sin(phi) ) ) ;
}
void RouletteSISLens::setXi( cv::Point2d x ) {
   xi = x ;
   double angle = atan2( x.y, x.x ) ;
   etaOffset = xi - einsteinR*cv::Point2d( cos(angle), sin(angle) ) ;
   etaOffset /= CHI ;
   etaOffset -= getEta() ;
}
