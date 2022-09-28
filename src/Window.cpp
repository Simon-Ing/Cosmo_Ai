/* (C) 2022: Hans Georg Schaathun <hg@schaathun.net> */

#include "Simulator.h"
#include <symengine/expression.h>
#include <symengine/lambda_double.h>
#include <thread>
#include <symengine/parser.h>
#include <fstream>

#define PI 3.14159265358979323846

double factorial_(unsigned int n);

Window::Window() :
        size(512),
        CHI_percent(50),
        einsteinR(size/20),
        sourceSize(size/20),
        xPosSlider(size*5 + 1),
        yPosSlider(size*5),
        mode(0), // 0 = point mass, 1 = sphere
        nterms(10)
{ 
}


void Window::initGui(){
    initSimulator( ) ;
    std::cout << "initGui\n" ;
    // Make the user interface and specify the function to be called when moving the sliders: update()
    cv::namedWindow("GL Simulator", cv::WINDOW_AUTOSIZE);
    cv::createTrackbar("Lens dist %    :", "GL Simulator", &CHI_percent, 100, updateXY, this);
    cv::createTrackbar("Einstein radius / Gamma:", "GL Simulator", &einsteinR, size, updateXY, this);
    cv::createTrackbar("Source sourceSize   :", "GL Simulator", &sourceSize, size / 10, updateSize, this);
    cv::createTrackbar("X position     :", "GL Simulator", &xPosSlider, 10*size, updateXY, this);
    cv::createTrackbar("Y position     :", "GL Simulator", &yPosSlider, 10*size, updateXY, this);
    cv::createTrackbar("\t\t\t\t\t\t\t\t\t\tMode, point/sphere:\t\t\t\t\t\t\t\t\t\t", "GL Simulator", &mode, 1, updateMode, this);

    cv::createTrackbar("sum from m=1 to...:", "GL Simulator", &nterms, 49, updateNterms, this);
    std::cout << "initGui DONE\n" ;
}

void Window::initSimulator(){
    std::cout << "initSimulator mode=" << mode << "\n" ;
    if ( NULL != sim ) delete sim ;
    if ( 0 == mode ) sim = new PointMassSimulator(size) ;
    else sim = new SphereSimulator(size) ;
    sim->setSource( new Source( size, sourceSize ) ) ;
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
    that->sim->updateXY( that->xPosSlider/10.0 - that->size/2.0, that->yPosSlider/10.0 - that->size/2.0,
                         that->CHI_percent / 100.0, that->einsteinR );
    /* The GUI has range 0..size; the simulator uses ±size/2. */
    that->drawImages() ;
}
void Window::updateSize(int, void* data){
    auto* that = (Window*)(data);
    that->sim->setSource( new Source( that->size, that->sourceSize ) ) ;
    that->sim->update() ;
    that->drawImages() ;
}
void Window::updateNterms(int, void* data){
    auto* that = (Window*)(data);
    that->sim->updateNterms( that->nterms ) ;
    that->drawImages() ;
}

void Window::drawImages() {
   cv::Mat imgActual = sim->getActual() ;
   cv::Mat imgDistorted = sim->getDistorted() ;

   // Copy both the actual and the distorted images into a new matDst array for display
   cv::Mat matDst(cv::Size(2*size, size), imgActual.type(), cv::Scalar::all(255));
   cv::Mat matRoi = matDst(cv::Rect(0, 0, size, size));
   imgActual.copyTo(matRoi);
   matRoi = matDst(cv::Rect(size, 0, size, size));
   imgDistorted.copyTo(matRoi);

   // Show the matDst array (i.e. both images) in the GUI window.
   cv::imshow("GL Simulator", matDst);
}

/*
void Window::drawImages2() {
   cv::Mat imgActual = sim->getActual() ;
   cv::Mat imgDistorted = sim->getDistorted() ;
   cv::Mat imgSecondary = sim->getSecondary() ;

   // Copy both the actual and the distorted images into a new matDst array for display
   cv::Mat matDst(cv::Size(2*size, size), imgActual.type(), cv::Scalar::all(255));
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
