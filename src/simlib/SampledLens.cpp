/* (C) 2023: Hans Georg Schaathun <georg@schaathun.net> */

/* This is only used as a base class for SampledPsiFunctionLens.  */

#include "cosmosim/Lens.h"
#include "simaux.h"

std::string SampledLens::idString() {
   return "SampledLens (Generic)" ;
};

void SampledLens::calculateAlphaBeta( cv::Point2d xi, int nterms ) {

    // Calculate all amplitudes for given X, Y, einsteinR

    int mp, m, s ;
    double C ;

    cv::Mat psi, matA, matB, matAouter, matBouter, matAx, matAy, matBx, matBy ;
    cv::Point2d ij ;

    psi = -this->getPsi() ;
    ij = imageCoordinate( xi, psi ) ;

    if (DEBUG) std::cout
              << "[SampledLens::calculateAlpaBeta] xi " << xi
              << " in pixel coordinates is "
              << ij << "; nterms=" << nterms << "\n" ;

    for ( mp = 0; mp <= nterms; mp++){
        s = mp+1 ; m = mp ;
        if ( mp == 0 ) {
          // This is the outer base case, for m=0, s=1
          gradient(psi, matBouter, matAouter) ;
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
        if (DEBUG) std::cout 
              << "SampledLens (" << m << ", " << s << ") alpha=" 
              << alphas_val[m][s]  << "; beta="
              << betas_val[m][s] << "\n"  ;

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
            if (DEBUG) std::cout 
              << "SampledLens (" << m << ", " << s << ") " 
              << alphas_val[m][s]  << "/"
              << betas_val[m][s] << "\n"  ;
        }
    }
}
double SampledLens::psiValue( double x, double y ) const { 
   cv::Point2d ij = imageCoordinate( cv::Point2d( x, y ), psi ) ;
   return psi.at<double>( ij ) ;
}
double SampledLens::psiXvalue( double x, double y ) const {
   cv::Point2d ij = imageCoordinate( cv::Point2d( x, y ), psi ) ;
   return psiY.at<double>( ij ) ;
}
double SampledLens::psiYvalue( double x, double y ) const { 
   cv::Point2d ij = imageCoordinate( cv::Point2d( x, y ), psi ) ;
   return psiX.at<double>( ij ) ;
}
cv::Mat SampledLens::getPsi() const {
   return psi ;
}
void SampledLens::updatePsi( ) { 
   return updatePsi( cv::Size(400,400) ) ;
}
void SampledLens::updatePsi( cv::Size size ) { 
   return ; 
}

double SampledLens::getAlpha( cv::Point2d xi, int m, int s ) {
   throw NotImplemented() ;
}
double SampledLens::getBeta( cv::Point2d xi, int m, int s ) {
   throw NotImplemented() ;
}

double SampledLens::getAlphaXi( int m, int s ) {
   return alphas_val[m][s] ;
}
double SampledLens::getBetaXi( int m, int s ) {
   return betas_val[m][s] ;
}

