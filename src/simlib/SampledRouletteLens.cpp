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
   cv::Point2d xi0, xi1 = chieta ;
   cv::Mat psi, psiX, psiY ;
   int cont = 1, count = 0, maxcount = 200 ;
   double dist, dist0=pow(10,12), threshold = 0.02 ;

   /* Get the lens potential */
   lens->updatePsi() ;
   psi = lens->getPsi() ;
   int ncols=psi.cols, nrows=psi.rows ;

   std::cout << "[SampledRouletteLens] updateApparentAbs()"
             << " chi*eta = " << chieta 
             << "; size: " << psi.size() << "\n" ;

   /** Differentiate the lens potential */
   gradient( -psi, psiX, psiY ) ;
   std::cout << "Types: " << psiX.type() << "/" << psiY.type() 
             << "/" << psi.type() << "\n" ;

   /* Diagnostic output */
   double minVal, maxVal;
   cv::Point minLoc, maxLoc;
   minMaxLoc( psiX, &minVal, &maxVal, &minLoc, &maxLoc ) ;
   std::cout << "[SampledRouletteLens] psiX min=" << minVal << "; max=" << maxVal << "\n" ;
   minMaxLoc( psiY, &minVal, &maxVal, &minLoc, &maxLoc ) ;
   std::cout << "[SampledRouletteLens] psiY min=" << minVal << "; max=" << maxVal << "\n" ;
   
   /** This block makes a fix-point iteration to find \xi. */
   while ( cont ) {
      xi0 = xi1 ;
      cv::Point2d ij = imageCoordinate( xi0, psi ) ;
      double x = -psiY.at<double>( ij ), y = -psiX.at<double>( ij ) ;
      std::cout << "[SampledRouletteLens] Fix pt it'n " << count
           << "; xi0=" << xi0 << "; Delta eta = " << x << ", " << y << "\n" ;
      xi1 = chieta + cv::Point2d( x, y ) ;
      dist = cv::norm( cv::Mat(xi1-xi0), cv::NORM_L2 ) ;
      if ( dist < threshold ) cont = 0 ;
      if ( ++count > maxcount ) cont = 0 ;
   }
   if ( dist > threshold ) {
      std::cout << "Bad approximation of xi: xi0=" << xi0 
            << "; xi1=" << xi1 << "; dist=" << dist 
            << "; nu=" << getNu() << "\n" ;
   } else {
      std::cout << "[SampledRouletteLens] Good approximation: xi0=" << xi0 
            << "; xi1=" << xi1 << "; nu=" <<  getNu() << "\n" ;
   }
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
