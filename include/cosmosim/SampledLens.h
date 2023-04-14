#ifndef SAMPLED_LENS_H
#define SAMPLED_LENS_H

#include "Simulator.h"
#include "Lens.h"

class PureSampledModel : public LensModel { 
public:
    using LensModel::LensModel ;
    PureSampledModel();
    PureSampledModel(bool);
protected:
    virtual void updateApparentAbs() ;
    virtual cv::Point2d calculateEta( cv::Point2d ) ;
    virtual void distort(int begin, int end, const cv::Mat& src, cv::Mat& dst) ;
    virtual cv::Point2d getDistortedPos(double r, double theta) const ;
    Lens *lens ;
private:
};

class PsiFunctionModel : public PureSampledModel { 
public:
    using PureSampledModel::PureSampledModel ;
    void setPsiFunctionLens( PsiFunctionLens* ) ;
protected:
    virtual void updateApparentAbs() ;
    virtual cv::Point2d calculateEta( cv::Point2d ) ;
private:
    PsiFunctionLens *psilens ;
};

#endif // SAMPLED_LENS_H
