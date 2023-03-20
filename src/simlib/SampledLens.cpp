/* (C) 2023: Hans Georg Schaathun <georg@schaathun.net> */

#include "cosmosim/Roulette.h"

#include <thread>

#include "simaux.h"

void SampledLens::calculateAlphaBeta() {

    // Calculate all amplitudes for given X, Y, einsteinR

    int mp, m, s ;
    double C ;
    cv::Mat psi = getPsi() ;
    cv::Mat matA, matB, matAouter, matBouter, matAx, matAy, matBx, matBy ;
    cv::Point2f ij = imageCoordinate( CHI*getNu(), psi ) ;

    std::cout << "[SampledLens] calculateAlphaBeta\n" ;

    for ( mp = 0; mp <= nterms; mp++){
        s = mp+1 ; m = mp ;
        if ( mp == 0 ) {
          // This is the outer base case, for m=0, s=1
          gradient(psi, matBouter, matAouter) ;
          matAouter *= -1 ;
          matBouter *= -1 ;
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


void SampledLens::updateApparentAbs( ) {
   cv::Point2f chieta = CHI*getEta() ;
   cv::Point2f xi0, xi1 = chieta ;
   cv::Mat psiX, psiY ;
   int cont = 1, count = 0, maxcount = 200 ;
   double dist, dist0=pow(10,12), threshold = 0.02 ;

   /* Get the lens potential */
   this->updatePsi() ;
   int ncols=psi.cols, nrows=psi.rows ;

   std::cout << "[SampledLens] updateApparentAbs()"
             << " chi*eta = " << chieta 
             << "; size: " << psi.size() << "\n" ;

   /** Differentiate the lens potential */
   gradient( psi, psiX, psiY ) ;
   std::cout << "Types: " << psiX.type() << "/" << psiY.type() 
             << "/" << psi.type() << "\n" ;

   /* Diagnostic output */
   double minVal, maxVal;
   cv::Point minLoc, maxLoc;
   minMaxLoc( psiX, &minVal, &maxVal, &minLoc, &maxLoc ) ;
   std::cout << "[SampledLens] psiX min=" << minVal << "; max=" << maxVal << "\n" ;
   minMaxLoc( psiY, &minVal, &maxVal, &minLoc, &maxLoc ) ;
   std::cout << "[SampledLens] psiY min=" << minVal << "; max=" << maxVal << "\n" ;
   
   /** This block performs a linear search for \xi.
    * It seems to work, but fix-point iteration should be faster
    * and allow for subpixel accuracy.
   for ( int i=0 ; i < nrows ; ++i ) {
      for ( int j=0 ; j < ncols ; ++j ) {
         cv::Point2f ij(i,j) ;
         cv::Point2f xy = pointCoordinate( ij, psi ) ;
         double x = psiY.at<double>( ij ), y = psiX.at<double>( ij ) ;
         cv::Point2f xitmp = chieta + cv::Point2f( x, y ) ;
         dist = cv::norm( cv::Mat(xitmp-xy), cv::NORM_L2 ) ;
         std::cout << "[SampledLens] (i,j)=(" << i << "," << j << ") xitmp= " 
                   << xitmp << "; dist=" << dist << "\n" ;
         if ( dist < dist0 ) {
            dist0 = dist ;
            xi0 = xitmp ;
            std::cout << "[SampledLens] xitmp= " << xitmp 
                      << "xy= " << xy << "; dist=" << dist0 << "\n" ;
         } 
      }
   }
   */

   /** This block makes a fix-point iteration to find \xi. */
   while ( cont ) {
      xi0 = xi1 ;
      cv::Point2f ij = imageCoordinate( xi0, psi ) ;
      double x = psiY.at<double>( ij ), y = psiX.at<double>( ij ) ;
      std::cout << "[SampledLens] Fix pt it'n " << count
           << "; xi0=" << xi0 << "; Delta eta = " << x << ", " << y << "\n" ;
      xi1 = chieta + cv::Point2f( x, y ) ;
      dist = cv::norm( cv::Mat(xi1-xi0), cv::NORM_L2 ) ;
      if ( dist < threshold ) cont = 0 ;
      if ( ++count > maxcount ) cont = 0 ;
   }
   nu = xi1/CHI ;
   if ( dist > threshold ) {
      std::cout << "Bad approximation of xi: xi0=" << xi0 
            << "; xi1=" << xi1 << "; dist=" << dist 
            << "; nu=" << nu << "\n" ;
   } else {
      std::cout << "[SampledLens] Good approximation: xi0=" << xi0 
            << "; xi1=" << xi1 << "; nu=" << nu << "\n" ;
   }
}
void SampledLens::updatePsi() { return ; }

double SampledLens::getNuAbs() const {
   return cv::norm( cv::Mat(nu), cv::NORM_L2 ) ;
}
cv::Point2f SampledLens::getNu() const {
   return nu ;
}

/* Getters for the images */
cv::Mat SampledLens::getActual() { 
   cv::Mat imgApparent = getApparent() ;

   cv::Mat imgActual 
        = cv::Mat::zeros(imgApparent.size(), imgApparent.type());

   cv::Mat tr = (cv::Mat_<double>(2,3) << 1, 0, getEta().x, 0, 1, -getEta().y);

   std::cout << "getActual() (x,y)=(" << getEta().x << "," << getEta().y << ")\n" ;

   cv::warpAffine(imgApparent, imgActual, tr, imgApparent.size()) ;
   return imgActual ; 
}

void SampledLens::update( cv::Mat imgApparent ) {

    auto startTime = std::chrono::system_clock::now();
    
    std::cout << "update() x=" << getEta().x << "; y= " << getEta().y 
              << "; R=" << getEtaAbs() << "; theta=" << phi
              << "; R_E=" << einsteinR << "; CHI=" << CHI << "\n" ;

    this->calculateAlphaBeta() ;

    // Make Distorted Image
    imgDistorted = cv::Mat::zeros(imgApparent.size(), imgApparent.type()) ;
    parallelDistort(imgApparent, imgDistorted);

    std::cout << "update() (x,y) = (" << getEta().x << ", " << getEta().y << ")\n" ;

    // Calculate run time for this function and print diagnostic output
    auto endTime = std::chrono::system_clock::now();
    std::cout << "Time to update(): " 
              << std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count() 
              << " milliseconds" << std::endl;

}
