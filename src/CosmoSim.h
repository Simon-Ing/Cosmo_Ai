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
                  CSIM_LENS_PM,
                  CSIM_NOLENS } ;

class CosmoSim {
private:
    int size=512, displaysize=512, basesize=512 ;
    double chi=0.5 ;
    int lensmode=CSIM_LENS_PM, oldlensmode=CSIM_NOLENS, einsteinR=20 ;
    int srcmode=CSIM_SOURCE_SPHERE, sourceSize=20, sourceSize2=10,
        sourceTheta=0 ;
    double xPos=10, yPos=0, rPos=10, thetaPos=0; ;
    int nterms=16 ;
    LensModel *sim = NULL ;
    Source *src = NULL ;
    bool running = false ;
    bool maskmode ;

    void initSource() ;
    void initLens() ;

public:
    CosmoSim();

    void setXY(double, double) ;
    void setPolar(int, int) ;
    void setCHI(int);
    void setNterms(int);
    // void updateDisplaySize(int);

    void setSourceMode(int);
    void setLensMode(int);
    void setEinsteinR(int);
    void setSourceParameters(int,int,int);

    bool runSim();
    void diagnostics();

    void maskImage() ;
    void showMask() ;
    void setMaskMode(bool) ;

    cv::Mat getApparent(bool) ;
    cv::Mat getActual(bool) ;
    cv::Mat getDistorted(bool) ;

};

#endif // COSMOSIM_FACADE_H
