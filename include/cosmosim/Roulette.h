#ifndef ROULETTE_H
#define ROULETTE_H

#include "Simulator.h"
#include "Lens.h"

#include <symengine/expression.h>
#include <symengine/lambda_double.h>

using namespace SymEngine;

#define PI 3.14159265358979323846


class RouletteAbstractLens : public LensModel { 
public:
    using LensModel::LensModel ;
protected:
    std::array<std::array<double, 202>, 201> alphas_val;
    std::array<std::array<double, 202>, 201> betas_val;

    virtual cv::Point2d getDistortedPos(double r, double theta) const;
    virtual void markMask( cv::InputOutputArray ) ;
    virtual void maskImage( cv::InputOutputArray, double ) ;
    virtual double getMaskRadius() const ;

    virtual void calculateAlphaBeta();
};

class RoulettePMLens : public RouletteAbstractLens { 
public:
    using RouletteAbstractLens::RouletteAbstractLens ;
protected:
    virtual cv::Point2d getDistortedPos(double r, double theta) const;
    virtual void updateApparentAbs() ;
    virtual void calculateAlphaBeta(); 
};

class SphereLens : public RouletteAbstractLens { 
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
class RouletteSISLens : public SphereLens { 
  protected:
    virtual void updateApparentAbs() ;
    virtual void setXi( cv::Point2d ) ;
  public:
    RouletteSISLens();
    RouletteSISLens(bool);
    RouletteSISLens(std::string,bool);
};


class SampledRouletteLens : public RouletteAbstractLens { 
public:
    SampledRouletteLens();
    SampledRouletteLens(bool);
    virtual void setLens( Lens* ) ;
protected:
    virtual void updateApparentAbs() ;
    virtual void setXi( cv::Point2d ) ;
};

class RouletteLens : public RouletteAbstractLens { 
  public:
    RouletteLens();
    RouletteLens(bool);
    virtual void setLens( Lens* ) ;

  protected:
    virtual void updateApparentAbs() ;
  private:

};

#endif // ROULETTE_H
