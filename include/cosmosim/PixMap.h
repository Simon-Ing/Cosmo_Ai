#ifndef PIXMAP_H
#define PIXMAP_H

#include "Simulator.h"

class LensMap { 

protected:
    cv::Mat psi, einsteinMap, massMap ;

private:

public:
    // void setEinsteinMap( cv::Mat ) ;
    void setPsi( cv::Mat ) ;
    void loadPsi( std::string ) ;
    cv::Mat getPsi( ) const ;
    cv::Mat getPsiImage( ) const ;
    cv::Mat getMassMap( ) const ;
    cv::Mat getMassImage() const ;
    cv::Mat getEinsteinMap( ) const ;
    virtual void updatePsi() ;
};

class PMCLens : public LensMap, public LensModel { 

public:
    using LensModel::LensModel ;

protected:
    virtual cv::Point2d getDistortedPos(double r, double theta) const;
    virtual void updateApparentAbs() ;

};
class PureSampledLens : public LensModel, public LensMap { 
public:
    using LensModel::LensModel ;
    PureSampledLens();
    PureSampledLens(bool);
protected:
    virtual void updateApparentAbs() ;
    cv::Point2d calculateEta( cv::Point2d ) ;
    void distort(int begin, int end, const cv::Mat& src, cv::Mat& dst) ;
    virtual cv::Point2d getDistortedPos(double r, double theta) const ;
};


#endif // PIXMAP_H
