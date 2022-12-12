/* (C) 2022: Hans Georg Schaathun <georg@schaathun.net> */

#ifndef COSMOSIM_FACADE_H
#define COSMOSIM_FACADE_H

#include "Simulator.h"
#include "Source.h"

class CosmoSim {
private:
    int size, displaysize, basesize;
    double chi = 0.5 ;
    int lensmode, einsteinR ;
    int srcmode, sourceSize, sourceSize2, sourceTheta ;
    int xPos, yPos, rPos, thetaPos; ;
    int nterms ;
    LensModel *sim = NULL ;
    Source *source = NULL ;

public:
    CosmoSim();

    void setXY(int, int) ;
    void setPolar(int, int) ;
    void setCHI(int);
    void setNterms(int);
    // void updateDisplaySize(int);

    void setLensMode(int);
    void setEinsteinR(int);
    void setSourceMode(int);
    void setSourceSize(int,int,int);

    cv::Mat getActual() ;
    cv::Mat getDistorted() ;

};

#endif // COSMOSIM_FACADE_H
