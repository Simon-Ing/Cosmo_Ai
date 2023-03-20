/* (C) 2023: Hans Georg Schaathun <georg@schaathun.net> */

#include "cosmosim/Roulette.h"
#include "simaux.h"

double SampledSISLens::psifunction( double x, double y ) {
   return (-einsteinR*sqrt( x*x + y*y )) ;
}

void SampledSISLens::updatePsi() { 
   cv::Mat im = getApparent() ;
   int nrows = im.rows ;
   int ncols = im.cols ;

   std::cout << "[SampledSISLens] updatePsi\n" ;

   psi = cv::Mat::zeros(im.size(), CV_64F );

   for ( int i=0 ; i<nrows ; ++i ) {
      for ( int j=0 ; j<ncols ; ++j ) {
         cv::Point2f ij( i, j ) ;
         cv::Point2f xy = pointCoordinate( ij, psi ) ;
	 psi.at<double>( ij ) = psifunction( xy.x, xy.y ) ;
      }
   }

   return ; 
}
