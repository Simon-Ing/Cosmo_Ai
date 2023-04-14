/* (C) 2023: Hans Georg Schaathun <georg@schaathun.net> */

#include "cosmosim/Lens.h"
#include "simaux.h"

#include <symengine/parser.h>
#include <fstream>

void Lens::updatePsi( ) { 
   return updatePsi( cv::Size(400,400) ) ;
}
void Lens::updatePsi( cv::Size size ) { 
   return ; 
}
void Lens::setEinsteinR( double r ) { einsteinR = r ; }

cv::Mat Lens::getPsi() const {
   return psi ;
}
cv::Mat Lens::getPsiX() const {
   return psiX ;
}
cv::Mat Lens::getPsiY() const {
   return psiY ;
}
cv::Mat Lens::getPsiImage() const {
   cv::Mat im, ps = getPsi() ;
   double minVal, maxVal;
   cv::Point minLoc, maxLoc;
   minMaxLoc( ps, &minVal, &maxVal, &minLoc, &maxLoc ) ;
   double s = std::max(-minVal,maxVal)*128 ;
   ps /= s ;
   ps += 128 ;
   ps.convertTo( im, CV_8S ) ;
   return im ;
}
cv::Mat Lens::getMassMap() const {
   cv::Mat psiX2, psiY2 ;
   Sobel(psi,psiX,CV_64FC1, 2, 0, 3, 1.0/8) ;
   Sobel(psi,psiY,CV_64FC1, 0, 2, 3, 1.0/8) ;
   return ( psiX + psiY ) / 2 ;
}
cv::Mat Lens::getMassImage() const {
   cv::Mat im, k ;
   double minVal, maxVal;
   cv::Point minLoc, maxLoc;

   k = getMassMap() ;
   minMaxLoc( k, &minVal, &maxVal, &minLoc, &maxLoc ) ;
   assert ( minVal >= 0 ) ;
   if ( maxVal > 255 ) {
     k /= maxVal ;
     k *= 255 ;
   }
   k.convertTo( im, CV_8S ) ;
   return im ;
}
cv::Mat Lens::getEinsteinMap() const {
   throw NotImplemented() ;
   // return einsteinMap ;
}

void Lens::setFile( std::string fn ) {
   std::cout << "[Lens.setFile()] " << fn << "\n" ;
   filename = fn ;
} 
void Lens::initAlphasBetas() {

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
            alphas_l[std::stoi(m)][std::stoi(s)].init({x, y, g}, *alpha_sym);
            betas_l[std::stoi(m)][std::stoi(s)].init({x, y, g}, *beta_sym);
        }
    }
}

double Lens::getAlpha( cv::Point2d xi, int m, int s ) {
   return alphas_l[m][s].call({xi.x, xi.y, einsteinR});
}
double Lens::getBeta( cv::Point2d xi, int m, int s ) {
   return betas_l[m][s].call({xi.x, xi.y, einsteinR});
}

void Lens::calculateAlphaBeta( cv::Point2d xi ) {
    std::cout << "[Lens.calculateAlphaBeta()] " << einsteinR << " - " << xi << "\n"  ;

    // calculate all amplitudes for given xi, einsteinR
    for (int m = 1; m <= nterms; m++){
        for (int s = (m+1)%2; s <= (m+1); s+=2){
            alphas_val[m][s] = alphas_l[m][s].call({xi.x, xi.y, einsteinR});
            betas_val[m][s] = betas_l[m][s].call({xi.x, xi.y, einsteinR});
        }
    }
}
double Lens::getAlphaXi( int m, int s ) {
   return alphas_val[m][s] ;
}
double Lens::getBetaXi( int m, int s ) {
   return betas_val[m][s] ;
}
double Lens::getXiAbs( double e ) {
   throw NotImplemented() ;
}
