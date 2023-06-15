/* (C) 2023: Hans Georg Schaathun <georg@schaathun.net> */

#include "cosmosim/Lens.h"
#include "simaux.h"

SampledPsiFunctionLens::SampledPsiFunctionLens( PsiFunctionLens *psilens ) {
   lens = psilens ;
}
void SampledPsiFunctionLens::setEinsteinR( double r ) {
   lens->setEinsteinR( einsteinR = r ) ; 
}

void SampledPsiFunctionLens::updatePsi( cv::Size size ) { 
   // cv::Mat im = getApparent() ;
   int nrows = size.height ;
   int ncols = size.width ;

   std::cout << "[SampledPsiFunctionLens] updatePsi " << size << "\n" ;

   lens->updatePsi(size) ;

   psi = lens->getPsi() ;

   gradient( -psi, psiX, psiY ) ;

   return ; 
}
