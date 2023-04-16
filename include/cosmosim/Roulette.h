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
    RouletteModel(bool);

    using LensModel::LensModel ;
    virtual void setLens( Lens* ) ;
protected:
    std::array<std::array<double, 202>, 201> alphas_val;
    std::array<std::array<double, 202>, 201> betas_val;

    virtual cv::Point2d getDistortedPos(double r, double theta) const;
    virtual void markMask( cv::InputOutputArray ) ;
    virtual void maskImage( cv::InputOutputArray, double ) ;
    virtual double getMaskRadius() const ;

    virtual void setXi( cv::Point2d ) ;

    virtual void updateApparentAbs() ;
    virtual void calculateAlphaBeta();
};

class RoulettePMLens : public RouletteModel { 
public:
    using RouletteModel::RouletteModel ;
protected:
    virtual cv::Point2d getDistortedPos(double r, double theta) const;
    virtual void updateApparentAbs() ;
    virtual void calculateAlphaBeta(); 
};

class SphereLens : public RouletteModel { 
  public:
    SphereLens();
    SphereLens(bool);
    SphereLens(std::string,bool);
    void setFile(std::string) ;
  protected:
    virtual void calculateAlphaBeta();
    virtual void updateApparentAbs() ;
  private:
    std::array<std::array<LambdaRealDoubleVisitor, 202>, 201> alphas_l;
    std::array<std::array<LambdaRealDoubleVisitor, 202>, 201> betas_l;

    std::string filename = "50.txt" ;
    void initAlphasBetas();
};


