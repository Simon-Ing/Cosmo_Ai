/* (C) 2023: Hans Georg Schaathun <georg@schaathun.net> */

#include "cosmosim/Roulette.h"

#include <symengine/expression.h>
#include <symengine/lambda_double.h>
#include <symengine/parser.h>

#include <thread>
#include <fstream>

SphereLens::SphereLens() :
   RouletteLens::RouletteLens()
{ 
    std::cout << "Instantiating SphereLens ... \n" ;
    initAlphasBetas();
}
SphereLens::SphereLens(bool centred) :
   RouletteLens::RouletteLens(centred)
{ 
    std::cout << "Instantiating SphereLens ... \n" ;
    initAlphasBetas();
}
SphereLens::SphereLens(std::string fn, bool centred) :
   filename(fn),
   RouletteLens::RouletteLens(centred)
{ 
    std::cout << "Instantiating SphereLens ... \n" ;
    initAlphasBetas();
}
void SphereLens::setFile( std::string fn ) {
    filename = fn ;
} 
void SphereLens::initAlphasBetas() {

    auto x = SymEngine::symbol("x");
    auto y = SymEngine::symbol("y");
    auto g = SymEngine::symbol("g");
    auto c = SymEngine::symbol("c");

    std::ifstream input;
    input.open(filename);

    if (!input.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }

    while (input) {
        std::string m, s;
        std::string alpha;
        std::string beta;
        std::getline(input, m, ':');
        std::getline(input, s, ':');
        std::getline(input, alpha, ':');
        std::getline(input, beta);
        if (input) {
            auto alpha_sym = SymEngine::parse(alpha);
            auto beta_sym = SymEngine::parse(beta);
            // The following two variables are unused.
            // SymEngine::LambdaRealDoubleVisitor alpha_num, beta_num;
            alphas_l[std::stoi(m)][std::stoi(s)].init({x, y, g}, *alpha_sym);
            betas_l[std::stoi(m)][std::stoi(s)].init({x, y, g}, *beta_sym);
        }
    }
}

void SphereLens::calculateAlphaBeta() {
    std::cout << "SphereLens calculateAlphaBeta\n" ;
    cv::Point2f xi = getNu()*CHI ;
    // double xiabs = getNuAbs()*CHI ;

    // calculate all amplitudes for given X, Y, einsteinR
    // This is done here to before the code is parallellised
    for (int m = 1; m <= nterms; m++){
        for (int s = (m+1)%2; s <= (m+1); s+=2){
            alphas_val[m][s] = alphas_l[m][s].call({xi.x, xi.y, einsteinR});
            betas_val[m][s] = betas_l[m][s].call({xi.x, xi.y, einsteinR});
        }
    }
}


void SphereLens::updateApparentAbs( ) {
    double r = getEtaAbs() + einsteinR/CHI ;
    nu = cv::Point2d( r, 0 ) ;
    // nu = cv::Point2d( r*cos(phi), r*sin(phi) ) ;
}
