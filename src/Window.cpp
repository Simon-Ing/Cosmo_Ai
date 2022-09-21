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
        size(300),
        CHI_percent(50),
        einsteinR(size/20),
        sourceSize(size/20),
        xPosSlider(size/2 + 1),
        yPosSlider(size/2),
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
    cv::createTrackbar("X position     :", "GL Simulator", &xPosSlider, size, updateXY, this);
    cv::createTrackbar("Y position     :", "GL Simulator", &yPosSlider, size, updateXY, this);
    cv::createTrackbar("\t\t\t\t\t\t\t\t\t\tMode, point/sphere:\t\t\t\t\t\t\t\t\t\t", "GL Simulator", &mode, 1, updateMode, this);

    cv::createTrackbar("sum from m=1 to...:", "GL Simulator", &nterms, 49, updateNterms, this);
    std::cout << "initGui DONE\n" ;
}

void Window::initSimulator(){
    std::cout << "initSimulator mode=" << mode << "\n" ;
    if ( NULL != sim ) delete sim ;
    if ( 0 == mode ) sim = new PointMassSimulator() ;
    else sim = new SphereSimulator() ;
    sim->updateAll( xPosSlider - size/2.0, yPosSlider - size/2.0,
         einsteinR, sourceSize, CHI_percent / 100.0, nterms ) ;
    std::cout << "initSimulator DONE\n" ;
}
void Window::updateMode(int, void* data){
    auto* that = (Window*)(data);
    that->initSimulator() ;
}
void Window::updateXY(int, void* data){
    auto* that = (Window*)(data);
    that->sim->updateXY( that->xPosSlider - that->size/2.0, that->yPosSlider - that->size/2.0,
                         that->CHI_percent / 100.0, that->einsteinR );
    /* The GUI has range 0..size; the simulator uses ±size/2. */
}
void Window::updateSize(int, void* data){
    auto* that = (Window*)(data);
    that->sim->updateSize( that->sourceSize ) ;
}
void Window::updateNterms(int, void* data){
    auto* that = (Window*)(data);
    that->sim->updateNterms( that->nterms ) ;
}
