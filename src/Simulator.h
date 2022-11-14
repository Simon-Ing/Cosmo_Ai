#ifndef SPHERICAL_SIMULATOR_H
#define SPHERICAL_SIMULATOR_H

#include "Source.h"
#include "opencv2/opencv.hpp"
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
    // double apparentX{};
    // double apparentY{};
    double actualAbs{};
    double apparentAbs{};
    double apparentAbs2{};

    cv::Mat imgApparent;
    cv::Mat imgDistorted;

    std::array<std::array<LambdaRealDoubleVisitor, 52>, 51> alphas_l;
    std::array<std::array<LambdaRealDoubleVisitor, 52>, 51> betas_l;
    std::array<std::array<double, 52>, 51> alphas_val;
    std::array<std::array<double, 52>, 51> betas_val;

public:
    LensModel();
    void update();

    virtual void updateXY(double, double, double, double) = 0;
    void updateNterms(int);
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
    void distort(int row, int col, const cv::Mat &src, cv::Mat &dst, int);
    void parallelDistort(const cv::Mat &src, cv::Mat &dst);

};

class PointMassLens : public LensModel { 
public:
    using LensModel::LensModel ;
    virtual void updateXY(double, double, double, double);
protected:
    virtual std::pair<double, double> getDistortedPos(double r, double theta) const;
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
    virtual void updateXY(double, double, double, double);
  protected:
    virtual void calculateAlphaBeta();
    virtual std::pair<double, double> getDistortedPos(double r, double theta) const;
  private:
    void initAlphasBetas();
};


class Window {
private:
    int mode;
    LensModel *sim = NULL ;
    int size;
    int CHI_percent;
    int sourceSize ;
    int einsteinR;
    int xPosSlider;
    int yPosSlider;
    int nterms;
    Source *source ;

public:
    Window();
    void initGui();

private:
    static void updateXY(int, void*);
    static void updateEinsteinR(int, void*);
    static void updateSize(int, void*);
    static void updateChi(int, void*);
    static void updateNterms(int, void*);
    static void updateMode(int, void*);

    void drawImages() ;
    /* void drawImages2() ; */
    void initSimulator();

};

/* simaux */
void refLines(cv::Mat&) ;

#endif //SPHERICAL_SIMULATOR_H
