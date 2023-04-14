/* (C) 2023: Hans Georg Schaathun <georg@schaathun.net> */

#include "cosmosim/Lens.h"
#include "simaux.h"

double SampledLens::getXiAbs( double e ) {
   throw NotImplemented() ;
}
void SampledLens::calculateAlphaBeta( cv::Point2d xi ) {

    // Calculate all amplitudes for given X, Y, einsteinR

    int mp, m, s ;
    double C ;
    this->updatePsi() ;
    cv::Mat psi = -this->getPsi() ;
    // std::cout << psi ;
    cv::Mat matA, matB, matAouter, matBouter, matAx, matAy, matBx, matBy ;
    cv::Point2d ij = imageCoordinate( xi, psi ) ;

    std::cout << "[SampledRouletteLens::calculateAlpaBeta] xi in image space is " << ij << 
       "; nterms=" << nterms << "\n" ;

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
