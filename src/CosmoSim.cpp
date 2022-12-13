#include "CosmoSim.h"
#include <pybind11/pybind11.h>
#include <opencv2/opencv.hpp>

enum SourceSpec { CSIM_SOURCE_SPHERE,
                  CSIM_SOURCE_ELLIPSE,
                  CSIM_SOURCE_TRIANGLE } ;

namespace py = pybind11;

void helloworld() {
   std::cout << "Hello World!\n" ;
   std::cout << "This is the CosmoSim Python Library!\n" ;
   return ;
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
   if ( src ) delete src ;
   switch ( srcmode ) {
       case CSIM_SOURCE_SPHERE:
         src = new SphericalSource( size, sourceSize ) ;
         break ;
       case CSIM_SOURCE_ELLIPSE:
         src = new EllipsoidSource( size, sourceSize,
               sourceSize2, sourceTheta*PI/180 ) ;
         break ;
       case CSIM_SOURCE_TRIANGLE:
         src = new TriangleSource( size, sourceSize, sourceTheta*PI/180 ) ;
         break ;
       default:
         std::cout << "No such mode!\n" ;
         exit(1) ;
    }
    if (sim) sim->setSource( src ) ;
} ;
cv::Mat CosmoSim::getActual() {
   return sim->getActual() ;
}
cv::Mat CosmoSim::getDistorted() {
   return sim->getDistorted() ;
}

PYBIND11_MODULE(CosmoSim, m) {
    m.doc() = "Wrapper for the CosmoSim simulator" ;

    m.def("helloworld", &helloworld, "A test function");

    py::class_<CosmoSim>(m, "CosmoSim")
        .def(py::init<>())
        .def("setLensMode", &CosmoSim::setLensMode)
        .def("setEinsteinR", &CosmoSim::setEinsteinR)
        .def("setSourceMode", &CosmoSim::setSourceMode)
        .def("setSourceSize", &CosmoSim::setSourceSize)
        .def("getActual", &CosmoSim::getActual)
        .def("getDistorted", &CosmoSim::getDistorted)
        ;

    // cv::Mat binding from https://alexsm.com/pybind11-buffer-protocol-opencv-to-numpy/
    pybind11::class_<cv::Mat>(m, "Image", pybind11::buffer_protocol())
        .def_buffer([](cv::Mat& im) -> pybind11::buffer_info {
            return pybind11::buffer_info(
                // Pointer to buffer
                im.data,
                // Size of one scalar
                sizeof(unsigned char),
                // Python struct-style format descriptor
                pybind11::format_descriptor<unsigned char>::format(),
                // Number of dimensions
                3,
                // Buffer dimensions
                { im.rows, im.cols, im.channels() },
                // Strides (in bytes) for each index
                {
                    sizeof(unsigned char) * im.channels() * im.cols,
                    sizeof(unsigned char) * im.channels(),
                    sizeof(unsigned char)
                }
            );
        });
}
