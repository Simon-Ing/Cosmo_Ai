/* (C) 2023: Hans Georg Schaathun <georg@schaathun.net> */

#include "cosmosim/Lens.h"
#include "simaux.h"

void PsiFunctionLens::updatePsi( cv::Size size ) { 
   // cv::Mat im = getApparent() ;
   int nrows = size.height ;
   int ncols = size.width ;

   std::cout << "[PsiFunctionLens] updatePsi\n" ;

   psi = cv::Mat::zeros(size, CV_64F );
   psiX = cv::Mat::zeros(size, CV_64F );
   psiY = cv::Mat::zeros(size, CV_64F );

   for ( int i=0 ; i<nrows ; ++i ) {
      for ( int j=0 ; j<ncols ; ++j ) {
         cv::Point2d ij( i, j ) ;
         cv::Point2d xy = pointCoordinate( ij, psi ) ;
	 psi.at<double>( ij ) = psifunction( xy.x, xy.y ) ;
	 // psiX.at<double>( ij ) = psiXfunction( xy.x, xy.y ) ;
	 // psiY.at<double>( ij ) = psiYfunction( xy.x, xy.y ) ;
      }
   }
   gradient( -psi, psiX, psiY ) ;

   return ; 
}
