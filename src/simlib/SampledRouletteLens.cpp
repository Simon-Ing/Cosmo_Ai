/* (C) 2023: Hans Georg Schaathun <georg@schaathun.net> */

#include "cosmosim/Roulette.h"

#include <thread>
#include "simaux.h"

SampledRouletteLens::SampledRouletteLens() :
   RouletteAbstractLens::RouletteAbstractLens()
{ 
    std::cout << "Instantiating SampledRouletteLens ... \n" ;
    rotatedMode = false ;
}
SampledRouletteLens::SampledRouletteLens(bool centred) :
   RouletteAbstractLens::RouletteAbstractLens(centred)
{ 
    std::cout << "Instantiating SampledRouletteLens ... \n" ;
    rotatedMode = false ;
}

void SampledRouletteLens::setLens( Lens *l ) {
   std::cout << "[SampledRouletteLens.setLens()]\n" ;
   lens = l ;
   lens->initAlphasBetas() ;
} 
void SampledRouletteLens::calculateAlphaBeta() {
    std::cout << "SampledRouletteLens calculateAlphaBeta\n" ;
    cv::Point2d xi = getXi() ;

    lens->calculateAlphaBeta( xi ) ;
}


void SampledRouletteLens::updateApparentAbs( ) {
   cv::Point2d chieta = CHI*getEta() ;
   cv::Point2d xi1 = lens->getXi( chieta) ;
   setNu( xi1/CHI ) ;
}
void SampledRouletteLens::setXi( cv::Point2d xi1 ) {
   cv::Point2d chieta, xy, ij ; 
   cv::Mat psi, psiX, psiY ;

   lens->updatePsi() ;
   psi = lens->getPsi() ;
   gradient( -psi, psiX, psiY ) ;
   ij = imageCoordinate( xi1, psi ) ;
   xy = cv::Point2d( -psiY.at<double>( ij ), -psiX.at<double>( ij ) );
   chieta = xi1 - xy ;

   xi = xi1 ;
   etaOffset = chieta/CHI - getEta() ;
}
