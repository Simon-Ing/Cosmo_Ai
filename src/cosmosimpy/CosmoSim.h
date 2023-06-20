/* (C) 2022: Hans Georg Schaathun <georg@schaathun.net> */

#ifndef COSMOSIM_FACADE_H
#define COSMOSIM_FACADE_H

#include "cosmosim/Roulette.h"
#include "cosmosim/Simulator.h"
#include "cosmosim/Source.h"
#include "cosmosim/Lens.h"

enum SourceSpec { CSIM_SOURCE_SPHERE,
                  CSIM_SOURCE_ELLIPSE,
                  CSIM_SOURCE_TRIANGLE } ;
enum ModelSpec { CSIM_MODEL_RAYTRACE,
                  CSIM_MODEL_ROULETTE,
                  CSIM_MODEL_POINTMASS_EXACT,
                  CSIM_MODEL_POINTMASS_ROULETTE,
                  CSIM_MODEL_SIS_ROULETTE,
                  CSIM_NOMODEL } ;
/*
enum LensSpec { CSIM_LENS_SPHERE,
                  CSIM_LENS_PM_ROULETTE, 
                  CSIM_LENS_ROULETTE_SIS, 
                  CSIM_LENS_PM,
                  CSIM_LENS_PURESAMPLED_SIS,
                  CSIM_LENS_SAMPLED_SIS,
                  CSIM_LENS_RAYTRACE,
                  CSIM_NOLENS } ;
                  */
enum PsiSpec    { CSIM_PSI_SIS,
                  CSIM_NOPSI_PM,
                  CSIM_NOPSI_SIS,
                  CSIM_NOPSI } ;

class CosmoSim {
private:
    int size=512, basesize=512 ;
    double chi=0.5 ;
    int modelmode=CSIM_MODEL_POINTMASS_EXACT, einsteinR=20 ;
    int sampledlens = 0, modelchanged = 0 ;
    int lensmode=CSIM_NOPSI_PM ;
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

    Lens *lens = NULL ;
    PsiFunctionLens *psilens = NULL ;

public:
    CosmoSim();

    void setFile(std::string) ;
    void setXY(double, double) ;
    void setPolar(int, int) ;
    void setCHI(double);
    void setNterms(int);
    void setImageSize(int);
    void setResolution(int);
    void setBGColour(int);

    void setSourceMode(int);
    void setModelMode(int);
    void setLensMode(int);
    void setSampled(int);
    void setEinsteinR(double);
    void setSourceParameters(double,double,double);

    bool runSim();
    bool moveSim( double, double ) ;
    void diagnostics();

    void maskImage(double) ;
    void showMask() ;
    void setMaskMode(bool) ;

    cv::Mat getSource(bool) ;
    cv::Mat getActual(bool) ;
    cv::Mat getDistorted(bool) ;

    cv::Mat getPsiMap() ;
    cv::Mat getMassMap() ;

    double getChi( ) ;
    double getAlpha( double x, double y, int m, int s ) ;
    double getBeta( double x, double y, int m, int s ) ;
    double getAlphaXi( int m, int s ) ;
    double getBetaXi( int m, int s ) ;

};

#endif // COSMOSIM_FACADE_H
