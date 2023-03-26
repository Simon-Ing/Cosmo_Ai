#ifndef ROULETTE_H
#define ROULETTE_H

#include "Simulator.h"
#include "PixMap.h"
#include "Lens.h"

#include <symengine/expression.h>
#include <symengine/lambda_double.h>

using namespace SymEngine;

#define PI 3.14159265358979323846

class RouletteLens : public LensModel { 
public:
    using LensModel::LensModel ;
protected:
    std::array<std::array<double, 202>, 201> alphas_val;
    std::array<std::array<double, 202>, 201> betas_val;

    virtual cv::Point2d getDistortedPos(double r, double theta) const;
    virtual void markMask( cv::InputOutputArray ) ;
    virtual void maskImage( cv::InputOutputArray, double ) ;
    virtual double getMaskRadius() const ;
};

class RoulettePMLens : public RouletteLens { 
public:
    using RouletteLens::RouletteLens ;
protected:
    virtual cv::Point2d getDistortedPos(double r, double theta) const;
    virtual void updateApparentAbs() ;
};

class SphereLens : public RouletteLens { 
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


class SampledRouletteLens : public RouletteLens, public LensMap { 
public:
    SampledRouletteLens();
    SampledRouletteLens(bool);
    void setLens( Lens * ) ;
protected:
    virtual void calculateAlphaBeta();
    virtual void updateApparentAbs() ;
    virtual void setXi( cv::Point2d ) ;
    virtual void updatePsi() ;
private:
    Lens *lens ;
};
class SampledSISLens : public SampledRouletteLens {
public:
    using SampledRouletteLens::SampledRouletteLens ;
protected:
    virtual void updatePsi() ;
private:
    double psifunction( double, double ) ;
};

#endif // ROULETTE_H
