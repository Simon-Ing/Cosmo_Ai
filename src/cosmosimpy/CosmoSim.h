/* (C) 2022: Hans Georg Schaathun <georg@schaathun.net> */

#ifndef COSMOSIM_FACADE_H
#define COSMOSIM_FACADE_H

#include "cosmosim/Roulette.h"
#include "cosmosim/Source.h"

enum SourceSpec { CSIM_SOURCE_SPHERE,
                  CSIM_SOURCE_ELLIPSE,
                  CSIM_SOURCE_TRIANGLE } ;
enum LensSpec { CSIM_LENS_SPHERE,
                  CSIM_LENS_ELLIPSE,
                  CSIM_LENS_SIS_ROULETTE, 
                  CSIM_LENS_PM_ROULETTE, 
                  CSIM_LENS_PM,
                  CSIM_LENS_SAMPLED,
                  CSIM_LENS_SAMPLED_SIS,
                  CSIM_NOLENS } ;

class CosmoSim {
private:
    int size=512, basesize=512 ;
    double chi=0.5 ;
    int lensmode=CSIM_LENS_PM, oldlensmode=CSIM_NOLENS, einsteinR=20 ;
    int srcmode=CSIM_SOURCE_SPHERE, sourceSize=20, sourceSize2=10,
        sourceTheta=0 ;
    double xPos=10, yPos=0, rPos=10, thetaPos=0; ;
    int nterms=16 ;
    int bgcolour=0 ;
    LensModel *sim = NULL ;
    Source *src = NULL ;
    bool running = false ;
    bool maskmode ;

    void initSource() ;
    void initLens() ;
    std::string filename = "50.txt" ;

public:
    CosmoSim();

    void setFile(std::string) ;
    void setXY(double, double) ;
    void setPolar(int, int) ;
    void setCHI(int);
    void setNterms(int);
    void setImageSize(int);
    void setResolution(int);
    void setBGColour(int);

    void setSourceMode(int);
    void setLensMode(int);
    void setEinsteinR(int);
    void setSourceParameters(int,int,int);

    bool runSim();
    void diagnostics();

    void maskImage() ;
    void showMask() ;
    void setMaskMode(bool) ;

    cv::Mat getSource(bool) ;
    cv::Mat getActual(bool) ;
    cv::Mat getDistorted(bool) ;

};

#endif // COSMOSIM_FACADE_H
