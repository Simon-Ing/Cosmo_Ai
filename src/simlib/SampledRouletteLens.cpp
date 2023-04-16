/* (C) 2023: Hans Georg Schaathun <georg@schaathun.net> */

#include "cosmosim/Roulette.h"

#include <thread>
#include "simaux.h"

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
