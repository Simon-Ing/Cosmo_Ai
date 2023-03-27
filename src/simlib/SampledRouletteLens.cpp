/* (C) 2023: Hans Georg Schaathun <georg@schaathun.net> */

#include "cosmosim/Roulette.h"

#include <thread>
#include "simaux.h"

SampledRouletteLens::SampledRouletteLens() :
   RouletteLens::RouletteLens()
{ 
    std::cout << "Instantiating SampledRouletteLens ... \n" ;
    rotatedMode = false ;
}
SampledRouletteLens::SampledRouletteLens(bool centred) :
   RouletteLens::RouletteLens(centred)
{ 
    std::cout << "Instantiating SampledRouletteLens ... \n" ;
    rotatedMode = false ;
}

void SampledRouletteLens::calculateAlphaBeta() {

    // Calculate all amplitudes for given X, Y, einsteinR

    int mp, m, s ;
    double C ;
    cv::Mat psi = -lens->getPsi() ;
    cv::Mat matA, matB, matAouter, matBouter, matAx, matAy, matBx, matBy ;
    cv::Point2d ij = imageCoordinate( getXi(), psi ) ;

    std::cout << "[SampledRouletteLens::calculateAlpaBeta] xi in image space is " << ij << "\n" ;

    for ( mp = 0; mp <= nterms; mp++){
        s = mp+1 ; m = mp ;
        if ( mp == 0 ) {
          // This is the outer base case, for m=0, s=1
          gradient(psi, matBouter, matAouter) ;
          // matAouter *= -1 ;
          // matBouter *= -1 ;
        } else {
          gradient(matAouter, matAy, matAx) ;
          gradient(matBouter, matBy, matBx) ;

          C = (m+1.0)/(m+1.0+s) ;
          //  if ( s == 1 ) C *= 2 ; // This is impossible, but used in the formula.

          matAouter = C*(matAx - matBy) ;
          matBouter = C*(matBx + matAy) ;
        }

        matA = matAouter.clone() ;
        matB = matBouter.clone() ;

        alphas_val[m][s] = matA.at<double>( ij ) ;
        betas_val[m][s] =  matB.at<double>( ij ) ;

        while( s > 0 && m < nterms ) {
            ++m ; --s ;
            C = (m+1.0)/(m+1.0-s) ;
            if ( s == 0 ) C /= 2.0 ;

            gradient(matA, matAy, matAx) ;
            gradient(matB, matBy, matBx) ;

            matA = C*(matAx + matBy) ;
            matB = C*(matBx - matAy) ;

            alphas_val[m][s] = matA.at<double>( ij ) ;
            betas_val[m][s] =  matB.at<double>( ij ) ;
        }
    }
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
         std::cout << "[SampledRouletteLens] (i,j)=(" << i << "," << j << ") xitmp= " 
                   << xitmp << "; dist=" << dist << "\n" ;
         if ( dist < dist0 ) {
            dist0 = dist ;
            xi0 = xitmp ;
            std::cout << "[SampledRouletteLens] xitmp= " << xitmp 
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
void SampledRouletteLens::setLens( Lens *l ) {
   lens = l ;
}
