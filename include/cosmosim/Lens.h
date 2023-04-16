#ifndef LENS_H
#define LENS_H

#if __has_include("opencv4/opencv2/opencv.hpp")
#include "opencv4/opencv2/opencv.hpp"
#else
#include "opencv2/opencv.hpp"
#endif

#include <symengine/expression.h>
#include <symengine/lambda_double.h>

using namespace SymEngine;

class Lens {

private:
    std::array<std::array<LambdaRealDoubleVisitor, 202>, 201> alphas_l;
    std::array<std::array<LambdaRealDoubleVisitor, 202>, 201> betas_l;

protected:
   double einsteinR ;
   std::string filename = "50.txt" ;
   cv::Mat psi, psiX, psiY, einsteinMap ;

    std::array<std::array<double, 202>, 201> alphas_val;
    std::array<std::array<double, 202>, 201> betas_val;
    int nterms=20;

public:
    virtual void updatePsi( cv::Size ) ;
    virtual void updatePsi( ) ;
    virtual void setEinsteinR( double ) ;

    cv::Mat getPsi( ) const ;
    cv::Mat getPsiX( ) const ;
    cv::Mat getPsiY( ) const ;
    cv::Mat getMassMap( ) const ;
    cv::Mat getEinsteinMap( ) const ; // Not implemented
    cv::Mat getPsiImage( ) const ;  // Discouraged
    cv::Mat getMassImage() const ;  // Discouraged


    void initAlphasBetas();
    virtual void calculateAlphaBeta( cv::Point2d xi );
    void setFile(std::string) ;
    void setNterms(int) ;

    std::array<std::array<double, 202>, 201> getAlphas( cv::Point2d xi ) ;
    std::array<std::array<double, 202>, 201> getBetas( cv::Point2d xi ) ;
    double getAlpha( cv::Point2d xi, int m, int s ) ;
    double getBeta( cv::Point2d xi, int m, int s ) ;
    double getAlphaXi( int m, int s ) ;
    double getBetaXi( int m, int s ) ;

    virtual double getXiAbs( double ) ;
    virtual cv::Point2d getXi( cv::Point2d ) ;
};

class SampledLens : public Lens {
public:
    virtual void calculateAlphaBeta( cv::Point2d xi );
    virtual cv::Point2d getXi( cv::Point2d ) ;
} ;

class PsiFunctionLens : public Lens {
public:
    virtual double psifunction( double, double ) = 0 ;
    virtual double psiXfunction( double, double ) = 0 ;
    virtual double psiYfunction( double, double ) = 0 ;
    virtual void updatePsi( cv::Size ) ;
} ;
class PixMapLens : public SampledLens {
public:
    void setPsi( cv::Mat ) ;
    void loadPsi( std::string ) ;
} ;

class SampledPsiFunctionLens : public SampledLens {
   private:
      PsiFunctionLens *lens ;
   public:
      SampledPsiFunctionLens(PsiFunctionLens*) ;
      virtual void updatePsi( cv::Size ) ;
      virtual void setEinsteinR( double ) ;
} ;

class SIS : public PsiFunctionLens { 

private:

public:
    virtual double psifunction( double, double ) ;
    virtual double psiXfunction( double, double ) ;
    virtual double psiYfunction( double, double ) ;

    virtual double getXiAbs( double ) ;
};


#endif // LENS_H
