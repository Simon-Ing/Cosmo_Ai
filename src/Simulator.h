#ifndef SPHERICAL_SIMULATOR_H
#define SPHERICAL_SIMULATOR_H

#include "opencv2/opencv.hpp"
#include <symengine/expression.h>
#include <symengine/lambda_double.h>

using namespace SymEngine;

#define PI 3.14159265358979323846

class Simulator {

protected:
    double CHI;
    int size;
    int sourceSize;
    double einsteinR;
    int nterms;

    double actualX{};
    double actualY{};
    // double apparentX{};
    // double apparentY{};
    double actualAbs{};
    double apparentAbs{};
    double apparentAbs2{};

    cv::Mat imgActual;
    cv::Mat imgApparent;
    cv::Mat imgDistorted;

    std::array<std::array<LambdaRealDoubleVisitor, 52>, 51> alphas_l;
    std::array<std::array<LambdaRealDoubleVisitor, 52>, 51> betas_l;
    std::array<std::array<double, 52>, 51> alphas_val;
    std::array<std::array<double, 52>, 51> betas_val;

public:
    Simulator();
    Simulator(int);
    void update();

    void updateXY(double, double, double, double);
    void updateSize(double);
    void updateNterms(int);
    void updateAll( double, double, double, double, double, int ) ;

    cv::Mat getActual() ;
    cv::Mat getApparent() ;
    cv::Mat getDistorted() ;
    cv::Mat getSecondary() ; // Made for testing

protected:
    virtual void calculateAlphaBeta();
    virtual std::pair<double, double> getDistortedPos(double r, double theta) const;

private:
    void distort(int row, int col, const cv::Mat &src, cv::Mat &dst);
    void parallelDistort(const cv::Mat &src, cv::Mat &dst);
    void drawParallel(cv::Mat &img, int xPos, int yPos);
    void drawSource(int begin, int end, cv::Mat &img, int xPos, int yPos);

};

class PointMassSimulator : public Simulator { 
  public:
    using Simulator::Simulator ;
};
class SphereSimulator : public Simulator { 
  public:
    SphereSimulator();
    SphereSimulator(int);
  protected:
    void calculateAlphaBeta();
    std::pair<double, double> getDistortedPos(double r, double theta) const;
  private:
    void initAlphasBetas();
};

class Window {
private:
    int mode;
    Simulator *sim = NULL ;
    int size;
    int CHI_percent;
    int sourceSize;
    int einsteinR;
    int xPosSlider;
    int yPosSlider;
    int nterms;

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
