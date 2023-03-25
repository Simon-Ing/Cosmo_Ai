/* (C) 2023: Hans Georg Schaathun <georg@schaathun.net> */

#include "cosmosim/Roulette.h"

#include <thread>
#include "simaux.h"

PureSampledLens::PureSampledLens() :
   LensModel::LensModel()
{ 
    std::cout << "Instantiating PureSampledLens ... \n" ;
    rotatedMode = false ;
}
PureSampledLens::PureSampledLens(bool centred) :
   LensModel::LensModel(centred)
{ 
    std::cout << "Instantiating PureSampledLens ... \n" ;
    rotatedMode = false ;
}


void PureSampledLens::updateApparentAbs( ) {
   cv::Point2d chieta = CHI*getEta() ;
   cv::Point2d xi0, xi1 = chieta ;
   cv::Mat psiX, psiY ;
   int cont = 1, count = 0, maxcount = 200 ;
   double dist, dist0=pow(10,12), threshold = 0.02 ;

   /* Get the lens potential */
   this->updatePsi() ;
   int ncols=psi.cols, nrows=psi.rows ;

   std::cout << "[PureSampledLens] updateApparentAbs()"
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
   std::cout << "[PureSampledLens] psiX min=" << minVal << "; max=" << maxVal << "\n" ;
   minMaxLoc( psiY, &minVal, &maxVal, &minLoc, &maxLoc ) ;
   std::cout << "[PureSampledLens] psiY min=" << minVal << "; max=" << maxVal << "\n" ;
   
   /** This block performs a linear search for \xi.
    * It seems to work, but fix-point iteration should be faster
    * and allow for subpixel accuracy.
   for ( int i=0 ; i < nrows ; ++i ) {
      for ( int j=0 ; j < ncols ; ++j ) {
         cv::Point2d ij(i,j) ;
         cv::Point2d xy = pointCoordinate( ij, psi ) ;
         double x = psiY.at<double>( ij ), y = psiX.at<double>( ij ) ;
         cv::Point2d xitmp = chieta + cv::Point2d( x, y ) ;
         dist = cv::norm( cv::Mat(xitmp-xy), cv::NORM_L2 ) ;
         std::cout << "[PureSampledLens] (i,j)=(" << i << "," << j << ") xitmp= " 
                   << xitmp << "; dist=" << dist << "\n" ;
         if ( dist < dist0 ) {
            dist0 = dist ;
            xi0 = xitmp ;
            std::cout << "[PureSampledLens] xitmp= " << xitmp 
                      << "xy= " << xy << "; dist=" << dist0 << "\n" ;
         } 
      }
   }
   */

   /** This block makes a fix-point iteration to find \xi. */
   while ( cont ) {
      xi0 = xi1 ;
      cv::Point2d ij = imageCoordinate( xi0, psi ) ;
      double x = -psiY.at<double>( ij ), y = -psiX.at<double>( ij ) ;
      std::cout << "[PureSampledLens] Fix pt it'n " << count
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
      std::cout << "[PureSampledLens] Good approximation: xi0=" << xi0 
            << "; xi1=" << xi1 << "; nu=" <<  getNu() << "\n" ;
   }
   setNu( xi1/CHI ) ;
}
void PureSampledLens::setXi( cv::Point2d xi1 ) {
   cv::Point2d chieta, xy, ij ; 
   cv::Mat psiX, psiY ;

   this->updatePsi() ;
   gradient( -psi, psiX, psiY ) ;
   ij = imageCoordinate( xi1, psi ) ;
   xy = cv::Point2d( -psiY.at<double>( ij ), -psiX.at<double>( ij ) );
   chieta = xi1 - xy ;

   xi = xi1 ;
   etaOffset = chieta/CHI - getEta() ;
}
void PureSampledLens::updatePsi() { return ; }
