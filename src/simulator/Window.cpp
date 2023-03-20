/* (C) 2022: Hans Georg Schaathun <georg@schaathun.net> */

/* Crude GUI written in the highlevel GUI API of OpenCV.
 * This is deprecated in favour of CosmoGUI.py written with
 * tkinter. */

/* DESIGN NOTES
 *
 * The `initGui` method sets up a window using the basic GUI functions 
 * from OpenCV.  This is crude, partly because the OpenCV API is only 
 * really meant for debugging, and partly because some features are
 * only available with QT and I have not yet figured out how to set
 * up OpenCV with QT support.
 *
 * When the GUI is initialised, `initSimulator` is called to
 * instantiate a `Simulator`.
 *
 * When a trackbar changes in the GUI, the corresponding callback
 * function is called; either `updateXY`, `updateMode`, `updateSize`,
 * or `updateNterms`.  Except for `updateMode`, each of these
 * call corresponding update functions in the simulator before they 
 * call `drawImages()` to get updated images from the simulator and 
 * display them.  When the mode changes, `updateMode` will instantiate
 * the relevant subclass of `Simulator`.
 *
 * The instance variables correspond to the trackbars in the GUI,
 * except for `size` which gives the image size.
 * This is constant, currently at 300, and has to be the same for
 * the `Window` and `Simulator` objects.
 */

#include "cosmosim/Simulator.h"
#include "cosmosim/Window.h"

#include <symengine/expression.h>
#include <symengine/lambda_double.h>
#include <symengine/parser.h>

#include <thread>
#include <fstream>

Window::Window() :
        size(512),
        basesize(512),
        displaysize(768),
        CHI_percent(50),
        einsteinR(size/20),
        sourceSize(size/20),
        sourceSize2(size/40),
        sourceTheta(90),
        xPosSlider(size*5 + 1),
        yPosSlider(size*5),
        rPosSlider(1),
        thetaPosSlider(0),
        mode(0), // 0 = point mass, 1 = sphere, 2 = roulette point mass
        srcmode(0), // 0 = sphere, 1 = ellipsoid, 2 = triangle 
        nterms(10)
{ 
}


void Window::initGui(){
    initSimulator( ) ;
    std::cout << "initGui\n" ;

    cv::namedWindow("GL Simulator", cv::WINDOW_AUTOSIZE);
    cv::createTrackbar("Lens dist %    :", "GL Simulator",
          &CHI_percent, 100, updateCHI, this);
    cv::createTrackbar("Einstein radius / Gamma:", "GL Simulator",
          &einsteinR, size, updateEinsteinR, this);
    cv::createTrackbar("Source Type         :", "GL Simulator",
          &srcmode, 2, updateSize, this);
    cv::createTrackbar("Source sourceSize   :", "GL Simulator",
          &sourceSize, size / 10, updateSize, this);
    cv::createTrackbar("Source sourceSize(2):", "GL Simulator",
          &sourceSize2, size / 10, updateSize, this);
    cv::createTrackbar("Source sourceTheta  :", "GL Simulator",
          &sourceTheta, 360, updateSize, this);
    cv::createTrackbar("X position     :", "GL Simulator",
          &xPosSlider, 10*size, updateXY, this);
    cv::createTrackbar("Y position     :", "GL Simulator",
          &yPosSlider, 10*size, updateXY, this);
    cv::createTrackbar("R position     :", "GL Simulator",
          &rPosSlider, 5*size, updatePolar, this);
    cv::createTrackbar("Phi position     :", "GL Simulator",
          &thetaPosSlider, 361, updatePolar, this);
    cv::createTrackbar("\t\t\t\t\t\t\t\t\t\tMode, point/sphere:\t",
          "GL Simulator", &mode, 2, updateMode, this);
    cv::createTrackbar("Resolution:", "GL Simulator",
          &basesize, size, updateDisplaySize, this);
    cv::createTrackbar("sum from m=1 to...:", "GL Simulator",
          &nterms, 49, updateNterms, this);
    std::cout << "initGui DONE\n" ;
}

void Window::initSimulator(){
    std::cout << "initSimulator mode=" << mode << "\n" ;
    if ( NULL != sim ) delete sim ;
    switch ( mode ) {
       case 0:
         sim = new PointMassLens() ;
         break ;
       case 1:
         sim = new SphereLens() ;
         break ;
       case 2:
         sim = new RoulettePMLens() ;
         break ;
       default:
         std::cout << "No such mode!\n" ;
         exit(1) ;
    }
    setSource() ;
    sim->updateAll( xPosSlider/10.0 - size/2.0, yPosSlider/10.0 - size/2.0,
         einsteinR, CHI_percent / 100.0, nterms ) ;
    drawImages() ;
    std::cout << "initSimulator DONE\n" ;
}
void Window::updateMode(int, void* data){
    auto* that = (Window*)(data);
    that->initSimulator() ;
    that->drawImages() ;
}
void Window::updateXY(int, void* data){
    auto* that = (Window*)(data);
    that->sim->updateXY( that->xPosSlider/10.0 - that->size/2.0,
          that->yPosSlider/10.0 - that->size/2.0,
                         that->CHI_percent / 100.0, that->einsteinR );
    /* The GUI has range 0..size; the simulator uses ±size/2. */
    that->drawImages() ;
}
void Window::updateEinsteinR(int, void* data){
    auto* that = (Window*)(data);
    that->sim->setEinsteinR( that->einsteinR ) ;
    that->sim->update() ;
    that->drawImages() ;
}
void Window::updateCHI(int, void* data){
    auto* that = (Window*)(data);
    that->sim->setCHI( that->CHI_percent / 100.0 ) ;
    that->sim->update() ;
    that->drawImages() ;
}
void Window::updatePolar(int, void* data){
    auto* that = (Window*)(data);
    that->sim->setPolar( that->rPosSlider/10.0,
                         that->thetaPosSlider,
                         that->CHI_percent / 100.0, that->einsteinR );
    /* The GUI has range 0..size; the simulator uses ±size/2. */
    that->sim->update() ;
    that->drawImages() ;
}
void Window::setSource(){
    Source *src ;
    switch ( srcmode ) {
       case 0:
         src = new SphericalSource( size, sourceSize ) ;
         break ;
       case 1:
         src = new EllipsoidSource( size, sourceSize,
               sourceSize2, sourceTheta*PI/180 ) ;
         break ;
       case 2:
         src = new TriangleSource( size, sourceSize, sourceTheta*PI/180 ) ;
         break ;
       default:
         std::cout << "No such mode!\n" ;
         exit(1) ;
    }
    sim->setSource( src ) ;
}
void Window::updateSize(int, void* data){
    auto* that = (Window*)(data);
    that->setSource() ;
    that->sim->update() ;
    that->drawImages() ;
}
void Window::updateNterms(int, void* data){
    auto* that = (Window*)(data);
    that->sim->updateNterms( that->nterms ) ;
    that->drawImages() ;
}
void Window::updateDisplaySize(int, void* data){
    auto* that = (Window*)(data);
    that->drawImages() ;
}

void Window::drawImages() {
   cv::Mat imgActual = sim->getActual() ;
   cv::Mat imgDistorted = sim->getDistorted() ;

   // This is a quick hack to prevent a crash on zero resolution.
   // It should be reviewed when a better GUI is designed.
   if ( basesize < 4 ) basesize = 4 ;

   // Copy both the actual and the distorted images into a new 
   // matDst array for display
   cv::Mat matDst(cv::Size(2*displaysize, displaysize), imgActual.type(),
                  cv::Scalar::all(255));
   cv::Mat matRoi = matDst(cv::Rect(0, 0, displaysize, displaysize));
   cv::resize(imgActual,matRoi,cv::Size(displaysize,displaysize) ) ;

   refLines( matRoi ) ;
   matRoi = matDst(cv::Rect(displaysize, 0, displaysize, displaysize));

   if ( basesize < size ) {
     cv::Mat tmp(cv::Size(basesize, basesize), imgActual.type(),
                  cv::Scalar::all(255));
     cv::resize(imgDistorted,tmp,cv::Size(basesize,basesize) ) ;
     cv::resize(tmp,matRoi,cv::Size(displaysize,displaysize),
             0, 0, cv::INTER_NEAREST ) ;
   } else {
     cv::resize(imgDistorted,matRoi,cv::Size(displaysize,displaysize) ) ;
   }
   refLines( matRoi ) ;

   // Show the matDst array (i.e. both images) in the GUI window.
   cv::imshow("GL Simulator", matDst);
}

/*
void Window::drawImages2() {
   cv::Mat imgActual = sim->getActual() ;
   cv::Mat imgDistorted = sim->getDistorted() ;
   cv::Mat imgSecondary = sim->getSecondary() ;

   // Copy both the actual and the distorted images into a new 
   // matDst array for display
   cv::Mat matDst(cv::Size(2*size, size), imgActual.type(),
                  cv::Scalar::all(255));
   cv::Mat matRoi = matDst(cv::Rect(0, 0, size, size));
   imgActual.copyTo(matRoi);
   matRoi = matDst(cv::Rect(size, 0, size, size));
   // imgDistorted.copyTo(matRoi);
   for ( int i=0 ; i < size ; ++i ) {
      for ( int j=0 ; j < size ; ++j ) {
         auto v1 = imgActual.at<uchar>(i, j);
         auto v2 = imgDistorted.at<uchar>(i, j);
         auto v3 = imgSecondary.at<uchar>(i, j);
         matRoi.at<uchar>(i, j) = { v1[0], v2[1], v3[1] } ;
      }
   }

   // Show the matDst array (i.e. both images) in the GUI window.
   cv::imshow("GL Simulator", matDst);
}
*/
