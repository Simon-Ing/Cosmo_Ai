#ifndef ROULETTE_H
#define ROULETTE_H

#include "Simulator.h"
#include "Lens.h"

#include <symengine/expression.h>
#include <symengine/lambda_double.h>

using namespace SymEngine;

#define PI 3.14159265358979323846


class RouletteModel : public LensModel { 
public:
    RouletteModel();

    using LensModel::LensModel ;
    virtual void setLens( Lens* ) ;
    // void setCentre( cv::Point2d ) ;
protected:
    std::array<std::array<double, 202>, 201> alphas_val;
    std::array<std::array<double, 202>, 201> betas_val;

    virtual cv::Point2d getDistortedPos(double r, double theta) const;
    // virtual void markMask( cv::InputOutputArray ) ;
    // virtual void maskImage( cv::InputOutputArray, double ) ;
    virtual double getMaskRadius() const ;

    virtual void setXi( cv::Point2d ) ;

    virtual void calculateAlphaBeta();
};


class RouletteRegenerator : public RouletteModel { 
  public:
    using RouletteModel::RouletteModel ;
    void setCentre( cv::Point2d, cv::Point2d ) ;
    void setAlphaXi( int, int, double ) ;
    void setBetaXi( int, int, double ) ;
  protected:
    virtual void updateApparentAbs() ;
    virtual void calculateAlphaBeta();
  private:
};


#endif /* ROULETTE_H */
