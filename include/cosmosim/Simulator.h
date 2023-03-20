#ifndef COSMOSIM_H
#define COSMOSIM_H

#include "Source.h"

#if __has_include("opencv4/opencv2/opencv.hpp")
#include "opencv4/opencv2/opencv.hpp"
#else
#include "opencv2/opencv.hpp"
#endif

#include <symengine/expression.h>
#include <symengine/lambda_double.h>

using namespace SymEngine;


#define PI 3.14159265358979323846

class LensModel {

protected:
    double CHI;
    Source *source ;
    double einsteinR;
    int nterms;

    int bgcolour = 0;

    double actualX{};
    double actualY{};
    double actualAbs{};
    double phi{};
    double apparentAbs{};
    double apparentAbs2{};
    bool maskMode = false ;
    double maskRadius = 1024*1024 ;

    // tentativeCentre is used as the shift when attempting 
    // to centre the distorted image in the image.
    double tentativeCentre = 0;

    cv::Mat imgApparent;
    cv::Mat imgDistorted;

    std::array<std::array<LambdaRealDoubleVisitor, 202>, 201> alphas_l;
    std::array<std::array<LambdaRealDoubleVisitor, 202>, 201> betas_l;
    std::array<std::array<double, 202>, 201> alphas_val;
    std::array<std::array<double, 202>, 201> betas_val;

private:
    bool centredMode = false ;

public:
    LensModel();
    LensModel(bool);
    ~LensModel();
    void update();
    void update( cv::Mat );
    void setCentred( bool ) ;
    void setMaskMode( bool ) ;
    void setBGColour( int ) ;

    void updateXY(double, double, double, double) ;
    void setXY(double, double, double, double) ;
    void setPolar(double, double, double, double) ;
    void setCHI(double) ;
    void setEinsteinR(double) ;
    virtual void updateApparentAbs()  = 0 ;
    virtual void maskImage( ) ;
    virtual void markMask( ) ;
    virtual void maskImage( cv::InputOutputArray ) ;
    virtual void markMask( cv::InputOutputArray ) ;
    void updateNterms(int);
    void setNterms(int);
    void updateAll( double, double, double, double, int ) ;
    void setSource(Source*) ;
    double getCentre() ;

    cv::Mat getActual() ;
    cv::Mat getApparent() ;
    cv::Mat getDistorted() ;
    cv::Mat getDistorted( double ) ;
    cv::Mat getSecondary() ; // Made for testing

protected:
    virtual void calculateAlphaBeta() ;
    virtual cv::Point2f getDistortedPos(double r, double theta) const = 0 ;
    void parallelDistort(const cv::Mat &src, cv::Mat &dst);

private:
    void distort(int row, int col, const cv::Mat &src, cv::Mat &dst);

};

class PointMassLens : public LensModel { 
public:
    using LensModel::LensModel ;
protected:
    virtual cv::Point2f getDistortedPos(double r, double theta) const;
    virtual void updateApparentAbs() ;
};

class RoulettePMLens : public PointMassLens { 
public:
    using PointMassLens::PointMassLens ;
protected:
    virtual cv::Point2f getDistortedPos(double r, double theta) const;
    virtual void markMask( cv::InputOutputArray ) ;
    virtual void updateApparentAbs() ;
};

class SphereLens : public LensModel { 
  public:
    SphereLens();
    SphereLens(bool);
    SphereLens(std::string,bool);
    void setFile(std::string) ;
  protected:
    virtual void maskImage( cv::InputOutputArray ) ;
    virtual void markMask( cv::InputOutputArray ) ;
    virtual void calculateAlphaBeta();
    virtual cv::Point2f getDistortedPos(double r, double theta) const;
    virtual void updateApparentAbs() ;
  private:
    std::string filename = "50.txt" ;
    void initAlphasBetas();
};



/* simaux */
void refLines(cv::Mat&) ;

class NotImplemented : public std::logic_error
{
public:
    NotImplemented() : std::logic_error("Function not yet implemented") { };
};

#endif // COSMOSIM_H
