#ifndef SAMPLED_LENS_H
#define SAMPLED_LENS_H

#include "Simulator.h"
#include "Lens.h"

class RaytraceModel : public LensModel { 
public:
    using LensModel::LensModel ;
    RaytraceModel();
    RaytraceModel(bool);
protected:
    virtual void updateApparentAbs() ;
    virtual cv::Point2d calculateEta( cv::Point2d ) ;
    virtual void distort(int begin, int end, const cv::Mat& src, cv::Mat& dst) ;
    virtual cv::Point2d getDistortedPos(double r, double theta) const ;
private:
};

#endif // SAMPLED_LENS_H
