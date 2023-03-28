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

#endif // PIXMAP_H
