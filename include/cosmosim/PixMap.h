#ifndef PIXMAP_H
#define PIXMAP_H

#include "Simulator.h"


class PixMapLens : public LensModel { 

protected:
    cv::Mat psi, einsteinMap, massMap ;

private:

public:
    using LensModel::LensModel ;

    void setEinsteinMap( cv::Mat ) ;
    cv::Mat getPsi( ) ;
    cv::Mat getMassMap( ) ;
    cv::Mat getEinsteinMap( ) ;

};

class PMCLens : public PixMapLens { 

public:
    using PixMapLens::PixMapLens ;

protected:
    virtual std::pair<double, double> getDistortedPos(double r, double theta) const;
    virtual void updateApparentAbs() ;

};

class RoulettePMCLens : public PixMapLens { 
public:
    using PixMapLens::PixMapLens ;
    /*
    RoulettePMCLens();
    RoulettePMCLens(bool);
    RoulettePMCLens(std::string,bool);
    */
protected:
    virtual void maskImage( cv::InputOutputArray ) ;
    virtual void markMask( cv::InputOutputArray ) ;
    virtual void calculateAlphaBeta();
    virtual std::pair<double, double> getDistortedPos(double r, double theta) const;
    virtual void updateApparentAbs() ;
private:
    void initAlphasBetas();
};

#endif // PIXMAP_H
