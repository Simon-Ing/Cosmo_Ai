#ifndef COSMOSIM_H
#define COSMOSIM_H

#include "Source.h"
#include "Lens.h"

#if __has_include("opencv4/opencv2/opencv.hpp")
#include "opencv4/opencv2/opencv.hpp"
#else
#include "opencv2/opencv.hpp"
#endif

#define PI 3.14159265358979323846

class LensModel {
private:
    cv::Point2d eta ;  // Actual position in the source plane
    cv::Point2d nu ;   // Apparent position in the source plane

    virtual void updateInner();


protected:
    cv::Mat imgDistorted;

    void parallelDistort(const cv::Mat &src, cv::Mat &dst);
    virtual void distort(int row, int col, const cv::Mat &src, cv::Mat &dst);

    cv::Point2d xi = cv::Point2d(0,0) ;   // Local origin in the lens plane
    cv::Point2d etaOffset = cv::Point2d(0,0) ;
        // Offset in the source plane resulting from moving xi
    double CHI;
    Source *source ;
    Lens *lens = NULL ;
    int nterms;
    double phi{};

    int bgcolour = 0;

    bool maskMode = false ;
    virtual double getMaskRadius() const ;
    void setNu( cv::Point2d ) ;
    virtual void setXi( cv::Point2d ) ;

    virtual void updateApparentAbs() ;
    virtual void calculateAlphaBeta() ;
    virtual cv::Point2d getDistortedPos(double r, double theta) const = 0 ;

public:
    LensModel();
    virtual ~LensModel();
    cv::Point2d getOffset( cv::Point2d ) ;
    cv::Point2d getRelativeEta( cv::Point2d ) ;
    void update();
    void update( cv::Point2d );
    void setCentred( bool ) ;
    void setMaskMode( bool ) ;
    void setBGColour( int ) ;

    double getNuAbs() const ;
    cv::Point2d getNu() const ;
    double getXiAbs() const ;
    cv::Point2d getXi() const ;
    cv::Point2d getTrueXi() const ;
    double getEtaAbs() const ;
    double getEtaSquare() const ;
    cv::Point2d getEta() const ;
    cv::Point2d getCentre() const ;

    void setXY(double, double) ;
    void setPolar(double, double) ;
    void setCHI(double) ;
    void setNterms(int);
    void setSource(Source*) ;

    virtual void maskImage( ) ;
    virtual void maskImage( double ) ;
    virtual void markMask( ) ;
    virtual void markMask( cv::InputOutputArray ) ;
    virtual void maskImage( cv::InputOutputArray, double ) ;

    cv::Mat getActual() const ;
    virtual cv::Mat getApparent() const ;
    cv::Mat getSource() const ;
    cv::Mat getDistorted() const ;

    virtual void setLens( Lens* ) ;
};

class RotatedModel : public LensModel { 
public:
   using LensModel::LensModel ;
protected:
    virtual cv::Mat getApparent() const ;
    virtual void updateInner();
};

class PointMassLens : public RotatedModel { 
public:
    using RotatedModel::RotatedModel ;
protected:
    virtual cv::Point2d getDistortedPos(double r, double theta) const;
    virtual void updateApparentAbs() ;
};


class RoulettePMLens : public RotatedModel { 
public:
    RoulettePMLens();
protected:
    virtual cv::Point2d getDistortedPos(double r, double theta) const;
    virtual void updateApparentAbs() ;
};


class RaytraceModel : public LensModel { 
public:
    using LensModel::LensModel ;
    RaytraceModel();
protected:
    virtual cv::Point2d calculateEta( cv::Point2d ) ;
    virtual void distort(int begin, int end, const cv::Mat& src, cv::Mat& dst) ;
    virtual cv::Point2d getDistortedPos(double r, double theta) const ;
private:
};

/* simaux */
void refLines(cv::Mat&) ;

class NotImplemented : public std::logic_error
{
public:
    NotImplemented() : std::logic_error("Function not yet implemented") { };
};
class NotSupported : public std::logic_error
{
public:
    NotSupported() : std::logic_error("Operation not supported in this context.") { };
};

#endif // COSMOSIM_H
