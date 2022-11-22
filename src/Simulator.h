#ifndef SPHERICAL_SIMULATOR_H
#define SPHERICAL_SIMULATOR_H

#include "Source.h"
#include "opencv4/opencv2/opencv.hpp"
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

    double actualX{};
    double actualY{};
    double actualAbs{};
    double phi{};
    double apparentAbs{};
    double apparentAbs2{};

    // tentativeCentre is used as the shift when attempting 
    // to centre the distorted image in the image.
    double tentativeCentre = 0;

    cv::Mat imgApparent;
    cv::Mat imgDistorted;

    std::array<std::array<LambdaRealDoubleVisitor, 52>, 51> alphas_l;
    std::array<std::array<LambdaRealDoubleVisitor, 52>, 51> betas_l;
    std::array<std::array<double, 52>, 51> alphas_val;
    std::array<std::array<double, 52>, 51> betas_val;

private:
    bool centredMode = false ;

public:
    LensModel();
    LensModel(bool);
    void update();
    void setCentred( bool ) ;

    void updateXY(double, double, double, double) ;
    void setPolar(double, double, double, double) ;
    void setCHI(double) ;
    void setEinsteinR(double) ;
    virtual void updateApparentAbs()  = 0 ;
    void updateNterms(int);
    void setNterms(int);
    void updateAll( double, double, double, double, int ) ;
    void setSource(Source*) ;

    cv::Mat getActual() ;
    cv::Mat getApparent() ;
    cv::Mat getDistorted() ;
    cv::Mat getDistorted( double ) ;
    cv::Mat getSecondary() ; // Made for testing

protected:
    virtual void calculateAlphaBeta() ;
    virtual std::pair<double, double> getDistortedPos(double r, double theta) const = 0 ;

private:
    void distort(int row, int col, const cv::Mat &src, cv::Mat &dst);
    void parallelDistort(const cv::Mat &src, cv::Mat &dst);

};

class PointMassLens : public LensModel { 
public:
    using LensModel::LensModel ;
protected:
    virtual std::pair<double, double> getDistortedPos(double r, double theta) const;
    virtual void updateApparentAbs() ;
};

class RoulettePMLens : public PointMassLens { 
public:
    using PointMassLens::PointMassLens ;
protected:
    virtual std::pair<double, double> getDistortedPos(double r, double theta) const;
};

class SphereLens : public LensModel { 
  public:
    SphereLens();
    SphereLens(bool);
  protected:
    virtual void calculateAlphaBeta();
    virtual std::pair<double, double> getDistortedPos(double r, double theta) const;
    virtual void updateApparentAbs() ;
  private:
    void initAlphasBetas();
};


class Window {
private:
    int mode, srcmode;
    LensModel *sim = NULL ;
    int size, displaysize, basesize;
    int CHI_percent;
    int sourceSize, sourceSize2, sourceTheta ;
    int einsteinR;
    int xPosSlider;
    int yPosSlider;
    int rPosSlider, thetaPosSlider ;
    int nterms;
    Source *source ;

public:
    Window();
    void initGui();

private:
    static void updateXY(int, void*);
    static void updatePolar(int, void*);
    static void updateEinsteinR(int, void*);
    static void updateSize(int, void*);
    static void updateCHI(int, void*);
    static void updateNterms(int, void*);
    static void updateDisplaySize(int, void*);
    static void updateMode(int, void*);

    void drawImages() ;
    /* void drawImages2() ; */
    void initSimulator();

};

/* simaux */
void refLines(cv::Mat&) ;

#endif //SPHERICAL_SIMULATOR_H
