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

    // calculate all amplitudes for given X, Y, einsteinR
    // This is done here to before the code is parallellised
    for (int m = 1; m <= nterms; m++){
        for (int s = (m+1)%2; s <= (m+1); s+=2){
            // alphas_val[m][s] = alphas_l[m][s].call({xi.x, xi.y, einsteinR});
            // betas_val[m][s] = betas_l[m][s].call({xi.x, xi.y, einsteinR});
        }
    }
}


void RouletteLens::updateApparentAbs( ) {
    double r = getEtaAbs() + einsteinR/CHI ;
    setNu( cv::Point2d( r, 0 ) ) ;
    // nu = cv::Point2d( r*cos(phi), r*sin(phi) ) ;
}
