/* (C) 2022: Hans Georg Schaathun <hg@schaathun.net> */

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


int main(int argc, char *argv[]) {

    LensModel *simulator ;

    // Set Defaults
    int nterms = 16, maskmode=1 ;
    int CHI_percent=50 ;
    int einsteinR=10, X=20, Y=0, sourceSize=10, sigma2 = 10 ;
    int phi = -1 ;
    int imgsize = 512 ;
    double theta = 0 ;
    double CHI ;
    std::string simname = "test", dirname = "./" ;
    int mode = 'p', srcmode = 's' ;
    int opt ;
    cv::Mat im ;
    Source *src ;

    while ( (opt = getopt(argc,argv,"D:L:S:M:N:s:2:t:T:n:X:E:I:r:Rx:y:Z:")) > -1 ) {
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
          case 'L': mode = optarg[0] ; break ;
          case 'S': srcmode = optarg[0] ; break ;
          case 'D': dirname = convertToString( optarg ) ; break ;
          case 'Z': imgsize = atoi(optarg) ; break ;
          case 'M': maskmode = atoi(optarg) ; break ;
          case 'N': simname = convertToString( optarg ) ; break ;
       }
    }

    CHI = CHI_percent/100.0 ;
    std::cout << "testmask (CosmoSim)\n" ;

    switch ( mode ) {
       case 'r':
         std::cout << "Running Roulette Point Mass Lens (mode=" << mode << ")\n" ;
         simulator = new RoulettePMLens() ;
         break ;
       case 's':
       default:
         std::cout << "Running SphereLens (mode=" << mode << ")\n" ;
         simulator = new SphereLens() ;
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
    switch ( maskmode ) {
       case 1:
          simulator->maskImage() ;
          break ;
       case 2:
          simulator->markMask() ;
          break ;
    }

    im = simulator->getDistorted() ;
    refLines(im) ;
    cv::imwrite( dirname + "mask-" + simname + ".png", im );

}
