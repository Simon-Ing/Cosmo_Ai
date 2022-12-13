/* (C) 2022: Hans Georg Schaathun <georg@schaathun.net> */

#ifndef COSMOSIM_FACADE_H
#define COSMOSIM_FACADE_H

#include "Simulator.h"
#include "Source.h"

enum SourceSpec { CSIM_SOURCE_SPHERE,
                  CSIM_SOURCE_ELLIPSE,
                  CSIM_SOURCE_TRIANGLE } ;
enum LensSpec { CSIM_LENS_SPHERE,
                  CSIM_LENS_ELLIPSE,
                  CSIM_LENS_PM_ROULETTE, 
                  CSIM_LENS_PM } ;

class CosmoSim {
private:
    int size=512, displaysize=512, basesize=512 ;
    double chi=0.5 ;
    int lensmode=CSIM_LENS_PM, einsteinR=20 ;
    int srcmode=CSIM_SOURCE_SPHERE, sourceSize=20, sourceSize2=10, sourceTheta=0 ;
    int xPos=10, yPos=0, rPos=10, thetaPos=0; ;
    int nterms=16 ;
    LensModel *sim = NULL ;
    Source *src = NULL ;

public:
    CosmoSim();

    void setXY(int, int) ;
    void setPolar(int, int) ;
    void setCHI(int);
    void setNterms(int);
    // void updateDisplaySize(int);

    void setLensMode(int);
    void initLens() ;
    void setEinsteinR(int);
    void setSourceParameters(int,int,int,int);
    void initSource() ;

    void init();
    void runSim();

    cv::Mat getActual() ;
    cv::Mat getDistorted() ;

};

#endif // COSMOSIM_FACADE_H
