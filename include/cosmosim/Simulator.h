#ifndef COSMOSIM_H
#define COSMOSIM_H

#include "Source.h"

#if __has_include("opencv4/opencv2/opencv.hpp")
#include "opencv4/opencv2/opencv.hpp"
#else
#include "opencv2/opencv.hpp"
#endif

#define PI 3.14159265358979323846

class LensModel {
private:
    cv::Point2d eta ;  // Actual position in the source plane
protected:
    double CHI;
    Source *source ;
    double einsteinR;
    int nterms;

    int bgcolour = 0;

    cv::Point2d nu ;   // Apparent position in the source plane
    double phi{};
    double apparentAbs2{};
    bool maskMode = false ;
    virtual double getMaskRadius() const ;


    // tentativeCentre is used as the shift when attempting 
    // to centre the distorted image in the image.
    double tentativeCentre = 0;

    cv::Mat imgApparent;
    cv::Mat imgDistorted;

private:
    bool centredMode = false ;

public:
    LensModel();
    LensModel(bool);
    ~LensModel();
    void update();
    virtual void update( cv::Mat );
    void setCentred( bool ) ;
    void setMaskMode( bool ) ;
    void setBGColour( int ) ;

    double getNuAbs() const ;
    cv::Point2d getNu() const ;
    double getEtaAbs() const ;
    double getEtaSquare() const ;
    cv::Point2d getEta() const ;


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

    virtual cv::Mat getActual() ;
    cv::Mat getApparent() ;
    cv::Mat getDistorted() ;
    cv::Mat getDistorted( double ) ;
    cv::Mat getSecondary() ; // Made for testing

protected:
    virtual void calculateAlphaBeta() ;
    virtual cv::Point2d getDistortedPos(double r, double theta) const = 0 ;
    void parallelDistort(const cv::Mat &src, cv::Mat &dst);

private:
    void distort(int row, int col, const cv::Mat &src, cv::Mat &dst);

};

class PointMassLens : public LensModel { 
public:
    using LensModel::LensModel ;
protected:
    virtual cv::Point2d getDistortedPos(double r, double theta) const;
    virtual void updateApparentAbs() ;
};

/* simaux */
void refLines(cv::Mat&) ;

class NotImplemented : public std::logic_error
{
public:
    NotImplemented() : std::logic_error("Function not yet implemented") { };
};

#endif // COSMOSIM_H
