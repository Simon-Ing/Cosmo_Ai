/* (C) 2023: Hans Georg Schaathun <georg@schaathun.net> */

#include "cosmosim/Lens.h"
#include "simaux.h"

SampledPsiFunctionLens::SampledPsiFunctionLens( PsiFunctionLens *psilens ) :
  SampledPsiFunctionLens(psilens, 512) {}
SampledPsiFunctionLens::SampledPsiFunctionLens( PsiFunctionLens *psilens, int size ) {
   lens = psilens ;
   this->updatePsi( cv::Size( size, size ) ) ;
}

void SampledPsiFunctionLens::updatePsi( cv::Size size ) { 

   int nrows = size.height ;
   int ncols = size.width ;

   if (DEBUG) std::cout << "[SampledPsiFunctionLens] updatePsi (size=" << size << ")\n" ;

   psi = cv::Mat::zeros(size, CV_64F );

   for ( int i=0 ; i<nrows ; ++i ) {
      for ( int j=0 ; j<ncols ; ++j ) {
         cv::Point2d ij( i, j ) ;
         cv::Point2d xy = pointCoordinate( ij, psi ) ;
	 psi.at<double>( ij ) = lens->psiValue( xy.x, xy.y ) ;
      }
   }

   gradient( psi, psiX, psiY ) ;

   if (DEBUG) std::cout << "[SampledPsiFunctionLens] (" << size << ") updatePsi() returns\n" ;
   return ; 
}
double SampledPsiFunctionLens::criticalXi( double phi ) const {
   return lens->criticalXi( phi ) ;
}
cv::Point2d SampledPsiFunctionLens::caustic( double phi ) const {
   return lens->caustic( phi ) ;
}
