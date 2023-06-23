/* (C) 2023: Hans Georg Schaathun <georg@schaathun.net> */

#include "cosmosim/Roulette.h"
#include "simaux.h"

#define alpha_(m,s)  ( NULL == this->lens ? alphas_val[m][s] : this->lens->getAlphaXi( m, s ) )
#define beta_(m,s)  ( NULL == this->lens ? betas_val[m][s] : this->lens->getBetaXi( m, s ) )

RouletteModel::RouletteModel() :
   LensModel::LensModel()
{ 
    std::cout << "Instantiating RouletteModel ... \n" ;
    rotatedMode = false ;
}
RouletteModel::RouletteModel(bool centred) :
   LensModel::LensModel(centred)
{ 
    std::cout << "Instantiating RouletteModel ... \n" ;
    rotatedMode = false ;
}

void RouletteModel::setLens( Lens *l ) {
   std::cout << "[RouletteModel.setLens()]\n" ;
   lens = l ;
   lens->initAlphasBetas() ;
} 

void RouletteModel::maskImage( cv::InputOutputArray imgD, double scale ) {
      std::cout << "RouletteModel::maskImage\n" ;
      cv::Mat imgDistorted = getDistorted() ;
      cv::Point2d origo = imageCoordinate( getCentre(), imgDistorted ) ;
      origo = cv::Point2d( origo.y, origo.x ) ;
      cv::Mat mask( imgD.size(), CV_8UC1, cv::Scalar(255) ) ;
      cv::Mat black( imgD.size(), imgD.type(), cv::Scalar(0) ) ;
      cv::circle( mask, origo, scale*getMaskRadius(), cv::Scalar(0), cv::FILLED ) ;
      black.copyTo( imgD, mask ) ;
}
void RouletteModel::markMask( cv::InputOutputArray imgD ) {
      std::cout << "RouletteModel::maskImage\n" ;
      cv::Mat imgDistorted = getDistorted() ;
      cv::Point2d origo = imageCoordinate( getCentre(), imgDistorted ) ;
      origo = cv::Point2d( origo.y, origo.x ) ;
      cv::circle( imgD, origo, getMaskRadius(), cv::Scalar(255), 1 ) ;
      cv::circle( imgD, origo, 3, cv::Scalar(0), 1 ) ;
      cv::circle( imgD, origo, 1, cv::Scalar(0), cv::FILLED ) ;
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

void RouletteModel::updateApparentAbs( ) {
   cv::Point2d chieta = CHI*getEta() ;
   lens->updatePsi() ;
   cv::Point2d xi1 = lens->getXi( chieta ) ;
   setNu( xi1/CHI ) ;
}
void RouletteModel::setXi( cv::Point2d xi1 ) {
   // xi1 is an alternative reference point \xi'
   xi = xi1 ;   // reset \xi

   // etaOffset is the difference between source point corresponding to the
   // reference point in the lens plane and the actual source centre
   etaOffset = getOffset( xi1 ) ;
}
/*
void RouletteModel::setCentre( cv::Point2d nu1 ) {
   // nu1 (\nu') is the centre point in the distorted image 
   // \xi'=\chi\nu' will be used as reference point in the lens plane.
   cv::Point2d chieta, eta1, ij ; 
   cv::Mat psi, psiX, psiY ;
   
   cv::Point2d xi1 = CHI*nu1 ; // \xi' = \chi\nu'

   lens->updatePsi() ;
   psi = lens->getPsi() ;
   psiX = lens->getPsiX() ;
   psiY = lens->getPsiY() ;
   ij = imageCoordinate( xi1, psi ) ;
   chieta = cv::Point2d( -psiY.at<double>( ij ), -psiX.at<double>( ij ) );

   setNu( cv::Point2d( 0,0 ) ) ;

   eta = chieta1/ETA ;
   etaOffset = -eta ;
}
*/
