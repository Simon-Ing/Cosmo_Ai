#include "CosmoSim.h"
#include <pybind11/pybind11.h>

enum SourceSpec { CSIM_SOURCE_SPHERE,
                  CSIM_SOURCE_ELLIPSE,
                  CSIM_SOURCE_TRIANGLE } ;

namespace py = pybind11;

int addT(int i, int j) {
    return i + j;
}

void CosmoSim::setCHI(int c) { chi = c/100 ; } ;
void CosmoSim::setNterms(int c) { nterms = c ; } ;
void CosmoSim::setXY(int x, int y) { xPos = x ; yPos = y ; } ;
void CosmoSim::setPolar(int r, int theta) { rPos = r ; thetaPos = theta ; } ;
void CosmoSim::setLensMode(int m) { lensmode = m ; } ;
void CosmoSim::setEinsteinR(int r) { einsteinR = r ; } ;
void CosmoSim::setSourceMode(int m) { srcmode = m ; } ;
void CosmoSim::setSourceSize(int s1, int s2, int theta ) {
   sourceSize = s1 ;
   sourceSize2 = s2 ;
   sourceTheta = theta ;
   if ( source ) delete source ;
   switch ( srcmode ) {
       case CSIM_SOURCE_SPHERE:
         source = new SphericalSource( size, sourceSize ) ;
         break ;
       case CSIM_SOURCE_ELLIPSE:
         source = new EllipsoidSource( size, sourceSize,
               sourceSize2, sourceTheta*PI/180 ) ;
         break ;
       case CSIM_SOURCE_TRIANGLE:
         source = new TriangleSource( size, sourceSize, sourceTheta*PI/180 ) ;
         break ;
       default:
         std::cout << "No such mode!\n" ;
         exit(1) ;
    }
    if (sim) sim->setSource( source ) ;
} ;

PYBIND11_MODULE(CosmoSim, m) {
    m.doc() = "Wrapper for the CosmoSim simulator" ;

    m.def("add", &addT, "A function that adds two numbers");

    py::class_<CosmoSim>(m, "CosmoSim")
        .def(py::init<>())
        .def("setLensMode", &CosmoSim::setLensMode)
        .def("setEinsteinR", &CosmoSim::setEinsteinR)
        .def("setSourceMode", &CosmoSim::setSourceMode)
        .def("setSourceSize", &CosmoSim::setSourceSize)
        ;

}
