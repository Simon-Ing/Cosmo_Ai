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

extern int debug ;
#ifndef DEBUG
#  define DEBUG debug
#endif

class SimulatorModel {
private:
    cv::Point2d eta ;  // Actual position in the source plane
    cv::Point2d nu ;   // Apparent position in the source plane

    cv::Point2d referenceXi = cv::Point2d(0,0) ;   // Local origin in the lens plane

protected:
    cv::Point2d etaOffset = cv::Point2d(0,0) ;
        // Offset in the source plane resulting from moving xi

    virtual void updateInner();
    cv::Mat imgDistorted;

    virtual void parallelDistort(const cv::Mat &src, cv::Mat &dst);
    virtual void distort(int row, int col, const cv::Mat &src, cv::Mat &dst);
    virtual void undistort(const cv::Mat &src, cv::Mat &dst);

    Source *source = NULL ;
    Lens *lens = NULL ;
    int nterms;
    double maskRadius=0 ;

    double getPhi() const ; // polar angle of source position

    int bgcolour = 0;

    bool maskMode = false ;
    double getMaskRadius() const ;
    void setNu( cv::Point2d ) ;
    void setXi( cv::Point2d ) ;

    virtual void updateApparentAbs() ;
    virtual void calculateAlphaBeta() ;
    virtual cv::Point2d getDistortedPos(double r, double theta) const ;

    double getXiAbs() const ;
    double getEtaSquare() const ;
    cv::Point2d getEta() const ;

public:
    SimulatorModel();
    virtual ~SimulatorModel();
    cv::Point2d getOffset( ) ;
    cv::Point2d getRelativeEta( cv::Point2d ) ;
    cv::Point2d getRelativeEtaPy( double, double ) ;

    void update();
    void update( cv::Point2d );

    /* Getters (images) */
    cv::Mat getActual() const ;
    virtual cv::Mat getApparent() const ;
    cv::Mat getDistorted() const ;

    /* Getters (Parameters) */
    cv::Point2d getNu() const ;
    cv::Point2d getXi() const ;

    /* Setters */
    virtual void setLens( Lens* ) ;

    void setXY(double, double) ;
    void setPolar(double, double) ;
    void setNterms(int);
    void setSource(Source*) ;
    // void setLightProfile( LightProfileSpec );

    void setCentred( bool ) ;
    void setMaskMode( bool ) ;
    void setBGColour( int ) ;

    /* Masking */
    void setMaskRadius( double ) ;
    void maskImage( ) ;
    void maskImage( double ) ;
    void markMask( ) ;
    void markMask( cv::InputOutputArray ) ;
    void maskImage( cv::InputOutputArray, double ) ;

    /* Critical Curves and Caustics */
    cv::Mat getCritical() ;
    void drawCritical( cv::Mat ) ;
    void drawCaustics( cv::Mat ) ;

};

class RaytraceModel : public SimulatorModel { 
public:
    using SimulatorModel::SimulatorModel ;
protected:
    virtual cv::Point2d calculateEta( cv::Point2d ) ;
    virtual void parallelDistort(const cv::Mat &src, cv::Mat &dst);
    virtual void distort(int begin, int end, const cv::Mat& src, cv::Mat& dst) ;
    virtual void undistort(const cv::Mat &src, cv::Mat &dst);
    virtual cv::Point2d getDistortedPos(double r, double theta) const ;
private:
};

/* simaux */
void refLines(cv::Mat&) ;

class NotImplemented : public std::exception
{
public:
    virtual const char * what () ;
};
class NotSupported : public std::exception
{
public:
    virtual const char * what () ;
};

#endif // COSMOSIM_H
