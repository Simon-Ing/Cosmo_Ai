/* (C) 2023: Hans Georg Schaathun <georg@schaathun.net> */

#include "cosmosim/Roulette.h"
#include "simaux.h"

void RouletteAbstractLens::maskImage( cv::InputOutputArray imgD, double scale ) {
      std::cout << "RouletteAbstractLens::maskImage\n" ;
      cv::Mat imgDistorted = getDistorted() ;
      cv::Point2d origo = imageCoordinate( getCentre(), imgDistorted ) ;
      origo = cv::Point2d( origo.y, origo.x ) ;
      cv::Mat mask( imgD.size(), CV_8UC1, cv::Scalar(255) ) ;
      cv::Mat black( imgD.size(), imgD.type(), cv::Scalar(0) ) ;
      cv::circle( mask, origo, scale*getMaskRadius(), cv::Scalar(0), cv::FILLED ) ;
      black.copyTo( imgD, mask ) ;
}
void RouletteAbstractLens::markMask( cv::InputOutputArray imgD ) {
      std::cout << "RouletteAbstractLens::maskImage\n" ;
      cv::Mat imgDistorted = getDistorted() ;
      cv::Point2d origo = imageCoordinate( getCentre(), imgDistorted ) ;
      origo = cv::Point2d( origo.y, origo.x ) ;
      cv::circle( imgD, origo, getMaskRadius(), cv::Scalar(255), 1 ) ;
      cv::circle( imgD, origo, 3, cv::Scalar(0), 1 ) ;
      cv::circle( imgD, origo, 1, cv::Scalar(0), cv::FILLED ) ;
}

// Calculate the main formula for the SIS model
cv::Point2d RouletteAbstractLens::getDistortedPos(double r, double theta) const {
    double nu1 = r*cos(theta) ;
    double nu2 = r*sin(theta) ;

    for (int m=1; m<=nterms; m++){
        double frac = pow(r, m) / factorial_(m);
        double subTerm1 = 0;
        double subTerm2 = 0;
        for (int s = (m+1)%2; s <= (m+1); s+=2){
            double alpha = alphas_val[m][s];
            double beta = betas_val[m][s];
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
    return cv::Point2d( nu1/CHI, nu2/CHI ) ;
}
double RouletteAbstractLens::getMaskRadius() const { 
   // Should this depend on the source position or the local origin?
   // return getNuAbs() ; 
   return getXiAbs()/CHI ; 
}
