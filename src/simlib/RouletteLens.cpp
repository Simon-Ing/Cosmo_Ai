/* (C) 2023: Hans Georg Schaathun <georg@schaathun.net> */

#include "cosmosim/Roulette.h"

#include <symengine/expression.h>
#include <symengine/lambda_double.h>
#include <symengine/parser.h>

#include <thread>
#include <fstream>

RouletteLens::RouletteLens() :
   RouletteAbstractLens::RouletteAbstractLens()
{ 
    std::cout << "Instantiating RouletteLens ... \n" ;
    rotatedMode = false ;
}
RouletteLens::RouletteLens(bool centred) :
   RouletteAbstractLens::RouletteAbstractLens(centred)
{ 
    std::cout << "Instantiating RouletteLens ... \n" ;
    rotatedMode = false ;
}

void RouletteLens::setLens( Lens *l ) {
    std::cout << "[RouletteLens.setLens()]\n" ;
    lens = l ;
    lens->initAlphasBetas() ;
} 

void RouletteLens::calculateAlphaBeta() {
    std::cout << "RouletteLens calculateAlphaBeta\n" ;
    cv::Point2d xi = getXi() ;

    lens->calculateAlphaBeta( xi ) ;
}


void RouletteLens::updateApparentAbs( ) {
    double r = lens->getXiAbs( getEtaAbs()*CHI )/CHI ;
    setNu( cv::Point2d( r*cos(phi), r*sin(phi) ) ) ;
}
