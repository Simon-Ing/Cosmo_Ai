/* (C) 2022: Hans Georg Schaathun <georg@schaathun.net> */

#include "CosmoSim.h"
#include <pybind11/pybind11.h>
#include <opencv2/opencv.hpp>


namespace py = pybind11;
CosmoSim::CosmoSim() {
   std::cout << "CosmoSim Constructor does nothing\n" ;
}

void helloworld() {
   std::cout << "Hello World!\n" ;
   std::cout << "This is the CosmoSim Python Library!\n" ;
}
void CosmoSim::diagnostics() {
   if ( src ) {
      cv::Mat im = src->getImage() ;
      std::cout << "Source Image " << im.rows << "x" << im.cols 
         << "x" << im.channels() << "\n" ;
   }
   if ( sim ) {
      cv::Mat im = sim->getDistorted() ;
      std::cout << "Distorted Image " << im.rows << "x" << im.cols 
         << "x" << im.channels() << "\n" ;
   }
   return ;
}


void CosmoSim::setRefLines(bool c) { refLinesMode = c ; }
void CosmoSim::setCHI(int c) { chi = c/100 ; }
void CosmoSim::setNterms(int c) { nterms = c ; }
void CosmoSim::setXY(int x, int y) { xPos = x ; yPos = y ; }
void CosmoSim::setPolar(int r, int theta) { rPos = r ; thetaPos = theta ; }
void CosmoSim::setLensMode(int m) { lensmode = m ; }
void CosmoSim::initLens() {
   bool centred = true ;
   if ( sim ) delete sim ;
   switch ( lensmode ) {
       case CSIM_LENS_SPHERE:
         std::cout << "Running SphereLens (mode=" << lensmode << ")\n" ;
         sim = new SphereLens(centred) ;
         break ;
       case CSIM_LENS_PM_ROULETTE:
         std::cout << "Running Roulette Point Mass Lens (mode=" << lensmode << ")\n" ;
         sim = new RoulettePMLens(centred) ;
         break ;
       case CSIM_LENS_PM:
         std::cout << "Running Point Mass Lens (mode=" << lensmode << ")\n" ;
         sim = new PointMassLens(centred) ;
         break ;
       default:
         std::cout << "No such lens mode!\n" ;
         throw NotImplemented();
    }
    return ;
}
void CosmoSim::setEinsteinR(int r) { einsteinR = r ; }
void CosmoSim::setSourceParameters(int mode, int s1, int s2, int theta ) {
   sourceSize = s1 ;
   sourceSize2 = s2 ;
   sourceTheta = theta ;
   srcmode = mode ;
}
void CosmoSim::initSource( ) {
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
         std::cout << "No such source mode!\n" ;
         throw NotImplemented();
    }
    if (sim) sim->setSource( src ) ;
}
void CosmoSim::runSim() { 
   if ( NULL == sim )
      throw std::bad_function_call() ;
   sim->update() ;
} 
cv::Mat CosmoSim::getActual() {
   if ( NULL == sim )
      throw std::bad_function_call() ;
   cv::Mat im = sim->getActual() ;
   if (refLinesMode) refLines(im) ;
   return im ;
}
cv::Mat CosmoSim::getDistorted() {
   if ( NULL == sim )
      throw std::bad_function_call() ;
   cv::Mat im = sim->getDistorted() ;
   if (refLinesMode) refLines(im) ;
   // It is necessary to clone because the distorted image is created
   // by cropping, and the pixmap is thus larger than the image,
   // causing subsequent conversion to a numpy array to be misaligned. 
   return im.clone() ;
}
void CosmoSim::init() {
   initLens() ;
   initSource() ;
   sim->setNterms( nterms ) ;
   sim->updateXY( xPos, yPos, chi, einsteinR ) ;
   return ;
}

PYBIND11_MODULE(CosmoSimPy, m) {
    m.doc() = "Wrapper for the CosmoSim simulator" ;

    m.def("helloworld", &helloworld, "A test function");

    py::class_<CosmoSim>(m, "CosmoSim")
        .def(py::init<>())
        .def("setLensMode", &CosmoSim::setLensMode)
        .def("setEinsteinR", &CosmoSim::setEinsteinR)
        .def("setNterms", &CosmoSim::setNterms)
        .def("setSourceParameters", &CosmoSim::setSourceParameters)
        .def("initLens", &CosmoSim::initSource)
        .def("initSource", &CosmoSim::initSource)
        .def("getActual", &CosmoSim::getActual)
        .def("getDistorted", &CosmoSim::getDistorted)
        .def("init", &CosmoSim::init)
        .def("runSim", &CosmoSim::runSim)
        .def("diagnostics", &CosmoSim::diagnostics)
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
    // Note.  The cv::Mat object returned needs to by wrapped in python:
    // `np.array(im, copy=False)` where `im` is the `Mat` object.

}
