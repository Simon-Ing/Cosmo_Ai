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
    int mode = 0 ;
    int opt ;

    while ( (opt = getopt(argc,argv,"SN:x:y:s:n:X:E:I:")) > -1 ) {
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
       }
    }

    std::cout << simname << "\n" ; 

    CHI = CHI_percent/100.0 ;

    if ( mode ) {
       std::cout << "Running SphereSimulator (mode=" << mode << ")\n" ;
       simulator = new SphereSimulator(imgsize) ;
    } else {
       std::cout << "Running Point Mass Simulator (mode=" << mode << ")\n" ;
       simulator = new PointMassSimulator(imgsize) ;
    }
    simulator->updateAll( X, Y, einsteinR, sourceSize, CHI, nterms );

    std::ostringstream filename;
    filename << CHI_percent << "," << einsteinR << "," << sourceSize << "," << X << "," << Y << ".png";
    cv::imwrite( "image-" + simname + filename.str(), simulator->getDistorted());
    cv::imwrite( "actual-" + simname + filename.str(), simulator->getActual());
    cv::imwrite( "secondary-" + simname + filename.str(), simulator->getSecondary());
}


