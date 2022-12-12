/* (C) 2022: Hans Georg Schaathun <georg@schaathun.net> */

#ifndef COSMOSIM_WINDOW_H
#define COSMOSIM_WINDOW_H

class Window {
private:
    int mode, srcmode;
    LensModel *sim = NULL ;
    int size, displaysize, basesize;
    int CHI_percent;
    int sourceSize, sourceSize2, sourceTheta ;
    int einsteinR;
    int xPosSlider;
    int yPosSlider;
    int rPosSlider, thetaPosSlider ;
    int nterms;
    Source *source ;

public:
    Window();
    void initGui();

private:
    static void updateXY(int, void*);
    static void updatePolar(int, void*);
    static void updateEinsteinR(int, void*);
    static void updateSize(int, void*);
    static void updateCHI(int, void*);
    static void updateNterms(int, void*);
    static void updateDisplaySize(int, void*);
    static void updateMode(int, void*);

    void drawImages() ;
    /* void drawImages2() ; */
    void initSimulator();
    void setSource();

};

#endif // COSMOSIM_WINDOW_H
