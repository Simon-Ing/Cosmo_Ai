/* (C) 2022: Hans Georg Schaathun <georg@schaathun.net> */

#include <unistd.h>
#include <string>
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

#define WRITE_APPARENT  1
#define WRITE_ACTUAL    2


int main(int argc, char *argv[]) {

    LensModel *simulator ;

    // Set Defaults
    int nterms = 16 ;
    int CHI_percent=50 ;
    int einsteinR=10, X=20, Y=0, sourceSize=10, sigma2 = 10 ;
    int phi = -1 ;
    int imgsize = 512 ;
    double theta = 0 ;
    double CHI ;
    std::string simname = "test", dirname = "./" ;
    int mode = 'p', srcmode = 's', refmode = 0 ;
    int opt ;
    cv::Mat im ;
    Source *src ;
    bool centred = false ;
    int outputmode = 0 ;

    while ( (opt = getopt(argc,argv,"aACD:L:S:N:s:2:t:T:n:X:E:I:r:Rx:y:Z:")) > -1 ) {
       switch(opt) {
          case 'x': X = atoi(optarg) ; break ;
          case 'y': Y = atoi(optarg) ; break ;
          case 'T': phi = atoi(optarg) ; break ;
          case 's': sourceSize = atoi(optarg) ; break ;
          case '2': sigma2 = atoi(optarg) ; break ;
          case 't': theta = PI*atoi(optarg)/180 ; break ;
          case 'X': CHI_percent = atoi(optarg) ; break ;
          case 'E': einsteinR = atoi(optarg) ; break ;
          case 'n': nterms = atoi(optarg) ; break ;
          case 'I': imgsize = atoi(optarg) ; break ;
          case 'N': simname = convertToString( optarg ) ; break ;
          case 'L': mode = optarg[0] ; break ;
          case 'S': srcmode = optarg[0] ; break ;
          case 'R': ++refmode ; break ;
          case 'C': centred = true ; break ;
          case 'D': dirname = convertToString( optarg ) ; break ;
          case 'Z': imgsize = atoi(optarg) ; break ;
          case 'a': outputmode |= WRITE_ACTUAL ; break ;
          case 'A': outputmode |= WRITE_APPARENT ; break ;
       }
    }

    CHI = CHI_percent/100.0 ;
    std::cout << "makeimage (CosmoSim)\n" ;

    switch ( mode ) {
       case 's':
         std::cout << "Running SphereLens (mode=" << mode << ")\n" ;
         simulator = new SphereLens(centred) ;
         break ;
       case 'r':
         std::cout << "Running Roulette Point Mass Lens (mode=" << mode << ")\n" ;
         simulator = new RoulettePMLens(centred) ;
         break ;
       case 'p':
       default:
         std::cout << "Running Point Mass Lens (mode=" << mode << ")\n" ;
         simulator = new PointMassLens(centred) ;
         break ;
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
    if ( phi < 0 ) {
        simulator->updateAll( X, Y, einsteinR, CHI, nterms );
    } else {
        simulator->setNterms( nterms ) ;
        simulator->setPolar( X, phi, CHI, einsteinR );
        simulator->update() ;
    }

    std::ostringstream filename;
    filename << ".png";

    im = simulator->getDistorted() ;
    if ( refmode ) refLines(im) ;
    cv::imwrite( dirname + "image-" + simname + filename.str(), im );

    if ( outputmode&WRITE_ACTUAL ) {
      im = simulator->getActual() ;
      std::cout << "Actual Image size " << im.rows << "x" << im.cols 
                << " - depth " << im.depth() << "\n" ;
      if ( refmode ) refLines(im) ; 
      cv::imwrite( dirname + "actual-" + simname + filename.str(), im );
    }

    if ( outputmode&WRITE_APPARENT ) {
       im = simulator->getApparent() ;
       if ( refmode ) refLines(im) ;
       std::cout << "Image size " << im.rows << "x" << im.cols 
                 << " - depth " << im.depth() << "\n" ;
       cv::imwrite( dirname + "apparent-" + simname + filename.str(), im );
    }
}




