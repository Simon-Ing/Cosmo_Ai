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

    Simulator *simulator ;

    // Set Defaults
    int nterms = 16 ;
    int CHI_percent=50 ;
    int einsteinR=10, X=20, Y=0, sourceSize=10 ;
    int imgsize = 512 ;
    double CHI ;
    std::string simname = "test" ;
    int mode = 0, refmode = 0 ;
    int opt ;
    cv::Mat im ;

    while ( (opt = getopt(argc,argv,"SN:x:y:s:n:X:E:I:L")) > -1 ) {
       if ( optarg ) {
          std::cout << "Option " << opt << " - " << optarg << "\n" ;
       } else {
          std::cout << "Option " << opt << "\n" ;
       }
       switch(opt) {
          case 'x': X = atoi(optarg) ; break ;
          case 'y': Y = atoi(optarg) ; break ;
          case 's': sourceSize = atoi(optarg) ; break ;
          case 'X': CHI_percent = atoi(optarg) ; break ;
          case 'E': einsteinR = atoi(optarg) ; break ;
          case 'n': nterms = atoi(optarg) ; break ;
          case 'I': imgsize = atoi(optarg) ; break ;
          case 'N': simname = convertToString( optarg ) ; break ;
          case 'S': ++mode ; break ;
          case 'L': ++refmode ; break ;
       }
    }

    std::cout << simname << "\n" ; 

    CHI = CHI_percent/100.0 ;

    if ( mode ) {
       std::cout << "Running SphereSimulator (mode=" << mode << ")\n" ;
       simulator = new SphereSimulator() ;
    } else {
       std::cout << "Running Point Mass Simulator (mode=" << mode << ")\n" ;
       simulator = new PointMassSimulator() ;
    }
    simulator->setSource( new Source( imgsize, sourceSize ) );
    simulator->updateAll( X, Y, einsteinR, CHI, nterms );

    std::ostringstream filename;
    filename << CHI_percent << "," << einsteinR << "," << sourceSize << "," << X << "," << Y << ".png";

    im = simulator->getDistorted() ;
    std::cout << "D1 Image size " << im.rows << "x" << im.cols << " - depth " << im.depth() << "\n" ;
    std::cout << "D1 Image type " << im.type() << "\n" ;
    if ( refmode ) refLines(im) ;
    std::cout << "D2 Image size " << im.rows << "x" << im.cols << " - depth " << im.depth() << "\n" ;
    std::cout << "D2 Image type " << im.type() << "\n" ;
    cv::imwrite( "image-" + simname + filename.str(), im );

    im = simulator->getActual() ;
    if ( refmode ) refLines(im) ;
    cv::imwrite( "actual-" + simname + filename.str(), im );
    im = simulator->getApparent() ;
    if ( refmode ) refLines(im) ;
    std::cout << "Image size " << im.rows << "x" << im.cols << " - depth " << im.depth() << "\n" ;
    std::cout << "Image type " << im.type() << "\n" ;
    cv::imwrite( "apparent-" + simname + filename.str(), im );

    im = simulator->getSecondary() ;
    std::cout << "Calculated Secondary image\n" ;
    std::cout << "Image size " << im.rows << "x" << im.cols << " - depth " << im.depth() << "\n" ;
    std::cout << "Image type " << im.type() << "\n" ;
    if ( refmode ) refLines(im) ;
    std::cout << "Added axes box\n" ;
    std::cout << "Image size " << im.rows << "x" << im.cols << " - depth " << im.depth() << "\n" ;
    std::cout << "Image type " << im.type() << "\n" ;
    cv::imwrite( "secondary-" + simname + filename.str(), im );
    std::cout << "Written to file\n" ;

    im = simulator->getApparent() ;
    if ( refmode ) refLines(im) ;
    cv::imwrite( "apparent2-" + simname + filename.str(), im );
}




