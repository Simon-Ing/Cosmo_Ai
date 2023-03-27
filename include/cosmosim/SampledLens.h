#ifndef SAMPLED_LENS_H
#define SAMPLED_LENS_H

#include "Simulator.h"
#include "Lens.h"

class PureSampledLens : public LensModel { 
public:
    using LensModel::LensModel ;
    PureSampledLens();
    PureSampledLens(bool);
    void setLens( Lens* ) ;
protected:
    virtual void updateApparentAbs() ;
    virtual cv::Point2d calculateEta( cv::Point2d ) ;
    virtual void distort(int begin, int end, const cv::Mat& src, cv::Mat& dst) ;
    virtual cv::Point2d getDistortedPos(double r, double theta) const ;
    cv::Mat psiX, psiY ;
    Lens *lens ;
private:
};

class PsiFunctionModel : public PureSampledLens { 
public:
    using PureSampledLens::PureSampledLens ;
protected:
    virtual void updateApparentAbs() ;
    virtual cv::Point2d calculateEta( cv::Point2d ) ;
    virtual void distort(int begin, int end, const cv::Mat& src, cv::Mat& dst) ;
private:
};

#endif // SAMPLED_LENS_H
