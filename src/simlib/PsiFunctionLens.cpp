/* (C) 2024: Hans Georg Schaathun <georg@schaathun.net> */

#include "cosmosim/Lens.h"
#include "simaux.h"

#include <symengine/parser.h>
#include <fstream>

std::string PsiFunctionLens::idString() {
   return "PsiFunctionLens" ;
};

void PsiFunctionLens::initAlphasBetas() {

    auto x = SymEngine::symbol("x");
    auto y = SymEngine::symbol("y");
    auto g = SymEngine::symbol("g"); /* Einstein Radius */
    auto f = SymEngine::symbol("f"); /* Ellipse Ratio */
    auto p = SymEngine::symbol("p"); /* theta  */

    std::ifstream input;
    if (DEBUG) std::cout << "[PsiFunctionLens.initAlphasBetas] Amplitudes file "
        << filename << "\n" ;

    if ( filename.compare("nosuchfile") == 0 )  return ;

    input.open(filename);

    if (!input.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    } else {
        if (DEBUG) std::cout 
           << "[initAlphasBetas] opened file " << filename << "\n" ;
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
            alphas_l[std::stoi(m)][std::stoi(s)].init({x, y, g, f, p}, *alpha_sym);
            betas_l[std::stoi(m)][std::stoi(s)].init({x, y, g, f, p}, *beta_sym);
        }
    }
}

void PsiFunctionLens::calculateAlphaBeta( cv::Point2d xi, int nterms ) {
    if (DEBUG) std::cout 
              << "[PsiFunctionLens.calculateAlphaBeta()] " << nterms << "; " 
              << einsteinR << " - " << xi << "\n"  ;

    // calculate all amplitudes for given xi, einsteinR
    for (int m = 1; m <= nterms; m++) {
        for (int s = (m+1)%2; s <= (m+1); s+=2) {
            alphas_val[m][s] = getAlpha( xi, m, s ) ;
            betas_val[m][s] = getBeta( xi, m, s ) ;
            // std::cout << "PsiFunctionLens (" << m << ", " << s << ") " << std::endl ;
              // << alphas_val[m][s]  << "/" << betas_val[m][s] << "\n"  ;
        }
    }
}

double PsiFunctionLens::getAlpha( cv::Point2d xi, int m, int s ) {
   double theta = orientation*PI/180 ;
   return alphas_l[m][s].call({xi.x, xi.y, einsteinR, ellipseratio, theta});
}
double PsiFunctionLens::getBeta( cv::Point2d xi, int m, int s ) {
   double theta = orientation*PI/180 ;
   return betas_l[m][s].call({xi.x, xi.y, einsteinR, ellipseratio, theta});
}

double PsiFunctionLens::getAlphaXi( int m, int s ) {
   return alphas_val[m][s] ;
}
double PsiFunctionLens::getBetaXi( int m, int s ) {
   return betas_val[m][s] ;
}

void PsiFunctionLens::setFile( std::string fn ) {
   filename = fn ;
   if (DEBUG) std::cout << "setFile " << filename << "\n" ;
} 

void PsiFunctionLens::setEinsteinR( double r ) { einsteinR = r ; }
double PsiFunctionLens::getEinsteinR() const { return einsteinR ; }
void PsiFunctionLens::setRatio( double r ) { 
   ellipseratio = r ; 
   if ( r >= 1 ) ellipseratio = 0.999 ;
   if ( r <= 0 ) ellipseratio = 0.001 ;
}
void PsiFunctionLens::setOrientation( double r ) { orientation = r ; }
double PsiFunctionLens::getOrientation() const { return this->orientation ; }
