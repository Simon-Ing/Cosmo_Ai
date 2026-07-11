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

#define MAXCLUSTER 50

class Lens {
public:

    virtual void calculateAlphaBeta( cv::Point2d, int ) ; /* Not implemented */

    virtual double getAlphaXi( int m, int s ) ; /* Not implemented */
    virtual double getBetaXi( int m, int s ) ; /* Not implemented */
    virtual double getAlphaPy( double, double, int m, int s ) ; 
    virtual double getBetaPy( double, double, int m, int s ) ; 
    virtual double getAlpha( cv::Point2d xi, int m, int s ) ; /* Not implemented */
    virtual double getBeta( cv::Point2d xi, int m, int s ) ; /* Not implemented */

    virtual cv::Point2d getXi( cv::Point2d ) ;

    virtual double psiValue( double, double ) const ; /* Not Implemented */
    virtual double psiXvalue( double, double ) const ; /* Not Implemented */
    virtual double psiYvalue( double, double ) const ; /* Not Implemented */

    virtual double criticalXi( double ) const ;
    virtual cv::Point2d caustic( double ) const ;

    virtual std::string idString() ;
};

class SampledLens : public Lens {
private:
    std::array<std::array<double, 202>, 201> alphas_val;
    std::array<std::array<double, 202>, 201> betas_val;
protected:
    cv::Mat psi, psiX, psiY ;
public:
    virtual void calculateAlphaBeta( cv::Point2d, int );

    virtual double getAlphaXi( int m, int s ) ;
    virtual double getBetaXi( int m, int s ) ;
    virtual double getAlpha( cv::Point2d xi, int m, int s ) ;
    virtual double getBeta( cv::Point2d xi, int m, int s ) ;

    virtual double psiValue( double, double ) const ;
    virtual double psiXvalue( double, double ) const ;
    virtual double psiYvalue( double, double ) const ;

    virtual void updatePsi( cv::Size ) ;
    virtual void updatePsi( ) ;
    cv::Mat getPsi( ) const ;

    virtual std::string idString() ;
} ;


class PsiFunctionLens : public Lens {
private:
    std::array<std::array<LambdaRealDoubleVisitor, 202>, 201> alphas_l;
    std::array<std::array<LambdaRealDoubleVisitor, 202>, 201> betas_l;
    std::array<std::array<double, 202>, 201> alphas_val;
    std::array<std::array<double, 202>, 201> betas_val;
protected:
    std::string filename = "nosuchfile" ;

    double einsteinR /* R_E or \xi_0 */,
           ellipseratio=1 /* f */,
	   orientation=0 /* \phi */ ;
public:
    virtual void initAlphasBetas();
    virtual void calculateAlphaBeta( cv::Point2d, int );

    void setFile(std::string) ;

    virtual double getAlphaXi( int m, int s ) ;
    virtual double getBetaXi( int m, int s ) ;
    virtual double getAlpha( cv::Point2d xi, int m, int s ) ;
    virtual double getBeta( cv::Point2d xi, int m, int s ) ;

    void setEinsteinR( double ) ;
    double getEinsteinR( ) const ;
    void setOrientation( double ) ;
    double getOrientation( ) const ;
    void setRatio( double ) ;

    virtual std::string idString() ;
} ;

class SampledPsiFunctionLens : public SampledLens {
   private:
      PsiFunctionLens *lens ;
   public:
      SampledPsiFunctionLens(PsiFunctionLens*) ;
      SampledPsiFunctionLens(PsiFunctionLens*,int) ;
      virtual void updatePsi( cv::Size ) ;
      virtual double criticalXi( double ) const ;
      virtual cv::Point2d caustic( double phi ) const ;

} ;

class PointMass : public PsiFunctionLens { 

private:

public:
    virtual double psiValue( double, double ) const ;
    virtual double psiXvalue( double, double ) const ;
    virtual double psiYvalue( double, double ) const ;

    virtual cv::Point2d getXi( cv::Point2d ) ;
    virtual std::string idString() ;
};
class SIS : public PsiFunctionLens { 

private:

public:
    virtual double psiValue( double, double ) const ;
    virtual double psiXvalue( double, double ) const ;
    virtual double psiYvalue( double, double ) const ;

    virtual cv::Point2d getXi( cv::Point2d ) ;

    virtual double criticalXi( double ) const ;
    virtual cv::Point2d caustic( double ) const ;
    virtual std::string idString() ;
};

class SIE : public PsiFunctionLens { 

private:
    double psifunctionPolar( double, double ) const ;
    // double psifunctionAligned( double, double ) const ;

public:
    virtual double psiValue( double, double ) const ;
    virtual double psiXvalue( double, double ) const ;
    virtual double psiYvalue( double, double ) const ;

    // virtual cv::Point2d getXi( cv::Point2d ) ;

    virtual double criticalXi( double ) const ;
    virtual cv::Point2d caustic( double ) const ;

    virtual std::string idString() ;
};

class ClusterLens : public PsiFunctionLens {
private:
    std::array<std::array<double, 202>, 201> alphas_val;
    std::array<std::array<double, 202>, 201> betas_val;
    PsiFunctionLens *lens[MAXCLUSTER] ;
    double xshift[MAXCLUSTER], yshift[MAXCLUSTER] ;
    int nlens = 0 ;
public:
    // virtual void addLens( PsiFunctionLens* );
    virtual void addLens( PsiFunctionLens*, double, double );
    virtual void initAlphasBetas();
    virtual void calculateAlphaBeta( cv::Point2d, int );

    virtual double psiValue( double, double ) const ;
    virtual double psiXvalue( double, double ) const ;
    virtual double psiYvalue( double, double ) const ;

    virtual double getAlpha( cv::Point2d xi, int m, int s ) ; 
    virtual double getBeta( cv::Point2d xi, int m, int s ) ; 
    virtual std::string idString() ;

    virtual double getAlphaXi( int m, int s ) ;
    virtual double getBetaXi( int m, int s ) ;
} ;

#endif // LENS_H
