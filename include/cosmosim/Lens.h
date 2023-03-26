#ifndef LENS_H
#define LENS_H

#include "cosmosim/PixMap.h"

class Lens {

private:
protected:
   double einsteinR ;
   cv::Mat psi, psiX, psiY ;

public:
    virtual double psifunction( double, double ) = 0 ;
    virtual double psiXfunction( double, double ) = 0 ;
    virtual double psiYfunction( double, double ) = 0 ;
    virtual void updatePsi( cv::Size ) ;
    virtual void updatePsi( ) ;
    void setEinsteinR( double ) ;
};

class SIS : public Lens { 

private:

public:
    virtual double psifunction( double, double ) ;
    virtual double psiXfunction( double, double ) ;
    virtual double psiYfunction( double, double ) ;
};


#endif // LENS_H
