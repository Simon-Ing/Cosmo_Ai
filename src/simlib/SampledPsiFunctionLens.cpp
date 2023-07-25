/* (C) 2023: Hans Georg Schaathun <georg@schaathun.net> */

#include "cosmosim/Lens.h"
#include "simaux.h"

SampledPsiFunctionLens::SampledPsiFunctionLens( PsiFunctionLens *psilens ) {
   lens = psilens ;
}
void SampledPsiFunctionLens::setEinsteinR( double r ) {
   lens->setEinsteinR( einsteinR = r ) ; 
}
void SampledPsiFunctionLens::setOrientation( double r ) {
   lens->setOrientation( phi = r ) ; 
}
void SampledPsiFunctionLens::setRatio( double r ) {
   lens->setRatio( ellipseratio = r ) ; 
}

void SampledPsiFunctionLens::updatePsi( cv::Size size ) { 
   std::cout << "[SampledPsiFunctionLens] updatePsi " << size << "\n" ;

   lens->updatePsi(size) ;
   psi = lens->getPsi() ;
   gradient( -psi, psiX, psiY ) ;

   return ; 
}
