#ifndef LENS_H
#define LENS_H

#if __has_include("opencv4/opencv2/opencv.hpp")
#include "opencv4/opencv2/opencv.hpp"
#else
#include "opencv2/opencv.hpp"
#endif

class Lens {

private:
protected:
   double einsteinR ;
   cv::Mat psi, psiX, psiY, einsteinMap ;

public:
    virtual void updatePsi( cv::Size ) ;
    virtual void updatePsi( ) ;
    void setEinsteinR( double ) ;

    cv::Mat getPsi( ) const ;
    cv::Mat getPsiX( ) const ;
    cv::Mat getPsiY( ) const ;
    cv::Mat getMassMap( ) const ;
    cv::Mat getEinsteinMap( ) const ; // Not implemented
    cv::Mat getPsiImage( ) const ;  // Discouraged
    cv::Mat getMassImage() const ;  // Discouraged

};
class PsiFunctionLens : public Lens {
public:
    virtual double psifunction( double, double ) = 0 ;
    virtual double psiXfunction( double, double ) = 0 ;
    virtual double psiYfunction( double, double ) = 0 ;
    virtual void updatePsi( cv::Size ) ;
} ;
class PixMapLens : public Lens {
public:
    void setPsi( cv::Mat ) ;
    void loadPsi( std::string ) ;
} ;

class SIS : public PsiFunctionLens { 

private:

public:
    virtual double psifunction( double, double ) ;
    virtual double psiXfunction( double, double ) ;
    virtual double psiYfunction( double, double ) ;
};


#endif // LENS_H