/* (C) 2022: Hans Georg Schaathun <hg@schaathun.net> *
 * Building on code by Simon Ingebrigtsen, Sondre Westbø Remøy,
 * Einar Leite Austnes, and Simon Nedreberg Runde
 */

#include <unistd.h>
#include <string>
#include <opencv2/opencv.hpp>
#include "Simulator.h"

int main(int argc, char *argv[]) {

    Simulator simulator;

    // Set Defaults
    int nterms = 16 ;
    int CHI_percent=50 ;
    int einsteinR=10, X=20, Y=0, sourceSize=10 ;
    int imgsize = 512 ;
    double CHI ;
    std::string simname = "test" ;
    int mode = 0 ;

    while ( int opt = getopt(argc,argv,"SN:x:y:s:n:X:E:I:") > -1 ) {
       switch(opt) {
          case 'x': X = atoi(optarg) ; break ;
          case 'y': Y = atoi(optarg) ; break ;
          case 's': sourceSize = atoi(optarg) ; break ;
          case 'X': CHI_percent = atoi(optarg) ; break ;
          case 'E': einsteinR = atoi(optarg) ; break ;
          case 'n': nterms = atoi(optarg) ; break ;
          case 'I': imgsize = atoi(optarg) ; break ;
          case 'N': simname = optarg ; break ;
          case 'S': ++mode ; break ;
       }
    }

    CHI = CHI_percent/100.0 ;

    if ( mode ) {
       simulator = SphereSimulator(imgsize) ;
    } else {
       simulator = PointMassSimulator(imgsize) ;
    }
    simulator.updateAll( X, Y, einsteinR, sourceSize, CHI, nterms );

    std::ostringstream filename;
    filename << CHI_percent << "," << einsteinR << "," << sourceSize << "," << X << "," << Y << ".png";
    cv::imwrite( "image- " + simname + filename.str(), simulator.getDistorted());
    cv::imwrite( "actual- " + simname + filename.str(), simulator.getActual());
}


