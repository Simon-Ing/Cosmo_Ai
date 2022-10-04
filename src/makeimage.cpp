/* (C) 2022: Hans Georg Schaathun <hg@schaathun.net> */

#include <unistd.h>
#include <string>
#include <opencv2/opencv.hpp>
#include "Simulator.h"

std::string convertToString(char*) ;

std::string convertToString(char* a)
{
    std::string s = "";
    while ( *a != 0 ) {
        s = s + *(a++) ;
    }
    std::cout << "convertToString: " << s << "\n" ;
    return s;
}


int main(int argc, char *argv[]) {

    LensModel *simulator ;

    // Set Defaults
    int nterms = 16 ;
    int CHI_percent=50 ;
    int einsteinR=10, X=20, Y=0, sourceSize=10, sigma2 = 10 ;
    double theta = 0 ;
    int imgsize = 512 ;
    double CHI ;
    std::string simname = "test" ;
    int mode = 0, srcmode = 0, refmode = 0 ;
    int opt ;
    cv::Mat im ;
    Source *src ;

    while ( (opt = getopt(argc,argv,"LS:N:x:y:s:2:t:n:X:E:I:R")) > -1 ) {
       if ( optarg ) {
          std::cout << "Option " << opt << " - " << optarg << "\n" ;
       } else {
          std::cout << "Option " << opt << "\n" ;
       }
       switch(opt) {
          case 'x': X = atoi(optarg) ; break ;
          case 'y': Y = atoi(optarg) ; break ;
          case 's': sourceSize = atoi(optarg) ; break ;
          case '2': sigma2 = atoi(optarg) ; break ;
          case 't': theta = PI*atoi(optarg)/180 ; break ;
          case 'X': CHI_percent = atoi(optarg) ; break ;
          case 'E': einsteinR = atoi(optarg) ; break ;
          case 'n': nterms = atoi(optarg) ; break ;
          case 'I': imgsize = atoi(optarg) ; break ;
          case 'N': simname = convertToString( optarg ) ; break ;
          case 'L': ++mode ; break ;
          case 'S': srcmode = optarg[0] ; break ;
          case 'R': ++refmode ; break ;
       }
    }

    std::cout << simname << "\n" ; 

    CHI = CHI_percent/100.0 ;

    if ( mode ) {
       std::cout << "Running SphereLens (mode=" << mode << ")\n" ;
       simulator = new SphereLens() ;
    } else {
       std::cout << "Running Point Mass Lens (mode=" << mode << ")\n" ;
       simulator = new PointMassLens() ;
    }
    switch ( srcmode ) {
       case 'e':
          std::cout << "Ellipsoid source, theta = " << theta << "\n" ;
          src = new EllipsoidSource( imgsize, sourceSize, sigma2, theta ) ;
          break ;
       case 't':
          std::cout << "Triangle source, theta = " << theta << "\n" ;
          src = new TriangleSource( imgsize, sourceSize, theta ) ;
          break ;
       case 's':
       default: 
          std::cout << "Spherical source\n" ;
          src = new SphericalSource( imgsize, sourceSize ) ;
          break ;


    }
    simulator->setSource( src ) ;
    simulator->updateAll( X, Y, einsteinR, CHI, nterms );

    std::ostringstream filename;
    filename << ".png";

    im = simulator->getDistorted() ;
    std::cout << "D1 Image size " << im.rows << "x" << im.cols << " - depth " << im.depth() << "\n" ;
    std::cout << "D1 Image type " << im.type() << "\n" ;
    if ( refmode ) refLines(im) ;
    std::cout << "D2 Image size " << im.rows << "x" << im.cols << " - depth " << im.depth() << "\n" ;
    std::cout << "D2 Image type " << im.type() << "\n" ;
    cv::imwrite( "image-" + simname + filename.str(), im );

    im = simulator->getActual() ;
    std::cout << "Actual Image size " << im.rows << "x" << im.cols << " - depth " << im.depth() << "\n" ;
    std::cout << "Actual Image type " << im.type() << "\n" ;
    if ( refmode ) refLines(im) ; // This does not work for some obscure reason
    cv::imwrite( "actual-" + simname + filename.str(), im );

    im = simulator->getSecondary() ;
    std::cout << "Calculated Secondary image\n" ;
    std::cout << "Image size " << im.rows << "x" << im.cols << " - depth " << im.depth() << "\n" ;
    std::cout << "Image type " << im.type() << "\n" ;
    if ( refmode ) refLines(im) ;
    std::cout << "Added axes box\n" ;
    cv::imwrite( "secondary-" + simname + filename.str(), im );
    std::cout << "Written to file\n" ;

    im = simulator->getApparent() ;
    if ( refmode ) refLines(im) ;
    std::cout << "Image size " << im.rows << "x" << im.cols << " - depth " << im.depth() << "\n" ;
    std::cout << "Image type " << im.type() << "\n" ;
    cv::imwrite( "apparent-" + simname + filename.str(), im );
}




