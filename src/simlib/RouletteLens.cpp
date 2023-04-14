/* (C) 2023: Hans Georg Schaathun <georg@schaathun.net> */

#include "cosmosim/Roulette.h"

#include <symengine/expression.h>
#include <symengine/lambda_double.h>
#include <symengine/parser.h>

#include <thread>
#include <fstream>

void RouletteLens::setLens( Lens *l ) {
   lens = l ;
   lens->initAlphasBetas() ;
} 

void RouletteLens::calculateAlphaBeta() {
    std::cout << "RouletteLens calculateAlphaBeta\n" ;
    cv::Point2d xi = getXi() ;

    lens->calculateAlphaBeta( xi ) ;
}


void RouletteLens::updateApparentAbs( ) {
    double r = lens->getXiAbs( getEtaAbs() ) + einsteinR/CHI ;
    setNu( cv::Point2d( r*cos(phi), r*sin(phi) ) ) ;
}
