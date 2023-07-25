/* (C) 2023: Hans Georg Schaathun <georg@schaathun.net> */

#include "cosmosim/Roulette.h"
#include "simaux.h"

#define alpha_(m,s)  ( NULL == this->lens ? alphas_val[m][s] : this->lens->getAlphaXi( m, s ) )
#define beta_(m,s)  ( NULL == this->lens ? betas_val[m][s] : this->lens->getBetaXi( m, s ) )

RouletteModel::RouletteModel() :
   LensModel::LensModel()
{ 
    std::cout << "Instantiating RouletteModel ... \n" ;
}

void RouletteModel::setLens( Lens *l ) {
   std::cout << "[RouletteModel.setLens()]\n" ;
   lens = l ;
   lens->initAlphasBetas() ;
} 

// Calculate the main formula for the SIS model
cv::Point2d RouletteModel::getDistortedPos(double r, double theta) const {
   // nu refers to a position in the source image relative to its centre.
   // It is scaled to the lens plane and rescaled when rpt is calculated below.
    double nu1 = r*cos(theta) ;
    double nu2 = r*sin(theta) ;

    for (int m=1; m<=nterms; m++){
        double frac = pow(r, m) / factorial_(m);
        double subTerm1 = 0;
        double subTerm2 = 0;
        for (int s = (m+1)%2; s <= (m+1); s+=2){
            double alpha = alpha_(m,s);
            double beta = beta_(m,s);
            double c_p = 1.0 + s/(m + 1.0);
            double c_m = 1.0 - s/(m + 1.0);
            subTerm1 += 0.5*( (alpha*cos((s-1)*theta) + beta*sin((s-1)*theta))*c_p 
                            + (alpha*cos((s+1)*theta) + beta*sin((s+1)*theta))*c_m );
            subTerm2 += 0.5*( (-alpha*sin((s-1)*theta) + beta*cos((s-1)*theta))*c_p 
                            + (alpha*sin((s+1)*theta) - beta*cos((s+1)*theta))*c_m);
        }
        nu1 += frac*subTerm1;
        nu2 += frac*subTerm2;
    }
    // The return value should be normalised coordinates in the source plane.
    // We have calculated the coordinates in the lens plane.
    // return cv::Point2d( nu1/CHI, nu2/CHI ) ;
    cv::Point2d rpt = cv::Point2d( nu1/CHI, nu2/CHI ) ;

    /*
    std::cout << "[getDistortedPos] (" << r << "," << theta << ") "
       << rpt << "\n" ;
    */
    return rpt ;
}
double RouletteModel::getMaskRadius() const { 
   // Should this depend on the source position or the local origin?
   // return getNuAbs() ; 
   return getXiAbs()/CHI ; 
}

void RouletteModel::calculateAlphaBeta() {
    cv::Point2d xi = getXi() ;

    std::cout << "RouletteModel calculateAlphaBeta ["
       << xi << "] ... \n" ;
    if ( lens == NULL ) throw NotSupported() ;

    lens->calculateAlphaBeta( xi ) ;
    std::cout << "RouletteModel calculateAlphaBeta done\n" ;
}

void RouletteModel::setXi( cv::Point2d xi1 ) {
   // xi1 is an alternative reference point \xi'
   xi = xi1 ;   // reset \xi

   // etaOffset is the difference between source point corresponding to the
   // reference point in the lens plane and the actual source centre
   etaOffset = getOffset( xi1 ) ;
}
