/* (C) 2022: Hans Georg Schaathun <georg@schaathun.net> */

#include "CosmoSim.h"

#include <pybind11/pybind11.h>
#include <opencv2/opencv.hpp>

namespace py = pybind11;

CosmoSim::CosmoSim() {
   std::cout << "CosmoSim Constructor\n" ;
   rPos = -1 ;
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

void CosmoSim::setFile( std::string fn ) {
    filename = fn ;
} 

void CosmoSim::setCHI(double c) { chi = c/100.0 ; }
void CosmoSim::setNterms(int c) { nterms = c ; }
void CosmoSim::setXY( double x, double y) { xPos = x ; yPos = y ; rPos = -1 ; }
void CosmoSim::setPolar(int r, int theta) { rPos = r ; thetaPos = theta ; }
void CosmoSim::setLensMode(int m) { lensmode = m ; }
void CosmoSim::setSourceMode(int m) { srcmode = m ; }
void CosmoSim::setMaskMode(bool b) { maskmode = b ; }
void CosmoSim::setBGColour(int b) { bgcolour = b ; }
void CosmoSim::initLens() {
   bool centred = false ;
   std::cout << "[CosmoSim.cpp] initLens\n" ;
   if ( lensmode == oldlensmode ) return ;
   if ( sim ) delete sim ;
   switch ( lensmode ) {
       case CSIM_LENS_SPHERE:
         std::cout << "Running SphereLens (mode=" << lensmode << ")\n" ;
         sim = new SphereLens(filename,centred) ;
         break ;
       case CSIM_LENS_SIS_ROULETTE:
         std::cout << "Running Roulette SIS Lens (mode=" << lensmode << ")\n" ;
         sim = new RouletteSISLens(filename,centred) ;
         break ;
       case CSIM_LENS_PM_ROULETTE:
         std::cout << "Running Roulette Point Mass Lens (mode=" << lensmode << ")\n" ;
         sim = new RoulettePMLens(centred) ;
         break ;
       case CSIM_LENS_PM:
         std::cout << "Running Point Mass Lens (mode=" << lensmode << ")\n" ;
         sim = new PointMassLens(centred) ;
         break ;
       case CSIM_LENS_PURESAMPLED:
         std::cout << "Running Pure Sampled Lens (mode=" << lensmode << ")\n" ;
         sim = new PureSampledLens(centred) ;
         break ;
       case CSIM_LENS_PURESAMPLED_SIS:
         std::cout << "Running Pure Sampled SIS Lens (mode=" << lensmode << ")\n" ;
         sim = new PureSampledSISLens(centred) ;
         break ;
       case CSIM_LENS_SAMPLED:
         std::cout << "Running Sampled Lens (mode=" << lensmode << ")\n" ;
         sim = new SampledLens(centred) ;
         break ;
       case CSIM_LENS_SAMPLED_SIS:
         std::cout << "Running Sampled SIS Lens (mode=" << lensmode << ")\n" ;
         sim = new SampledSISLens(centred) ;
         break ;
       default:
         std::cout << "No such lens mode!\n" ;
         throw NotImplemented();
    }
    oldlensmode = lensmode ;
    return ;
}
void CosmoSim::setEinsteinR(double r) { einsteinR = r ; }
void CosmoSim::setImageSize(int sz ) { size = sz ; }
void CosmoSim::setResolution(int sz ) { 
   basesize = sz ; 
   std::cout << "[setResolution] basesize=" << basesize << "; size=" << size << "\n" ;
}
void CosmoSim::setSourceParameters(double s1, double s2, double theta ) {
   sourceSize = s1 ;
   if ( s2 >= 0 ) sourceSize2 = s2 ;
   if ( theta >= 0 ) sourceTheta = theta ;
   // srcmode = mode ;
}
void CosmoSim::initSource( ) {
   std::cout << "[CosmoSim.cpp] initSource()\n" ;
   // Deleting the source object messes up the heap and causes
   // subsequent instantiation to fail.  This is probably because
   // the imgApparent (cv:;Mat) is not freed correctly.
   // if ( src ) delete src ;
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
    std::cout << "[CosmoSim.cpp] initSource() completes\n" ;
}
bool CosmoSim::runSim() { 
   std::cout << "[CosmoSim.cpp] runSim()\n" ;
   if ( running ) return false ;
   std::cout << "[CosmoSim.cpp] runSim() - running similator\n" ;
   initLens() ;
   initSource() ;
   sim->setBGColour( bgcolour ) ;
   sim->setNterms( nterms ) ;
   sim->setMaskMode( maskmode ) ;
   if ( rPos < 0 ) {
      sim->setXY( xPos, yPos, chi, einsteinR ) ;
   } else {
      sim->setPolar( rPos, thetaPos, chi, einsteinR ) ;
   }
   std::cout << "[runSim] set parameters, ready to run\n" ;
   Py_BEGIN_ALLOW_THREADS
   sim->update() ;
   Py_END_ALLOW_THREADS
   std::cout << "[CosmoSim.cpp] runSim() - complete\n" ;
   return true ;
}
bool CosmoSim::moveSim( double rot, double scale ) { 
   cv::Point2d xi = sim->getTrueXi(), xi1 ;
   xi1 = cv::Point2d( 
           xi.x*cos(rot) - xi.y*sin(rot),
           xi.x*sin(rot) + xi.y*cos(rot)
         );
   xi1 *= scale ;
   Py_BEGIN_ALLOW_THREADS
   sim->update( xi1 ) ;
   Py_END_ALLOW_THREADS
   return true ;
}
cv::Mat CosmoSim::getSource(bool refLinesMode) {
   if ( NULL == sim )
      throw std::bad_function_call() ;
   cv::Mat im = sim->getSource() ;
   if (refLinesMode) {
      im = im.clone() ;
      refLines(im) ;
   }
   return im ;
}
cv::Mat CosmoSim::getActual(bool refLinesMode) {
   if ( NULL == sim )
      throw std::bad_function_call() ;
   cv::Mat im = sim->getActual() ;
   std::cout << "basesize=" << basesize << "; size=" << size << "\n" ;
   if ( basesize < size ) {
      cv::Mat ret(cv::Size(basesize, basesize), im.type(),
                  cv::Scalar::all(255));
      cv::resize(im,ret,cv::Size(basesize,basesize) ) ;
      im = ret ;
   } else {
      im = im.clone() ;
   }
   if (refLinesMode) {
      refLines(im) ;
   }
   return im ;
}
void CosmoSim::maskImage( double scale ) {
          sim->maskImage( scale ) ;
}
void CosmoSim::showMask() {
          sim->markMask() ;
}

cv::Mat CosmoSim::getDistorted(bool refLinesMode) {
   if ( NULL == sim )
      throw std::bad_function_call() ;
   cv::Mat im ;
   if ( basesize < size ) {
      std::cout << "basesize=" << basesize << "; size=" << size << "\n" ;
      im = sim->getDistorted() ;
      cv::Mat ret(cv::Size(basesize, basesize), sim->getActual().type(),
                  cv::Scalar::all(255));
      cv::resize(im,ret,cv::Size(basesize,basesize) ) ;
      im = ret ;
   } else {
      // It is necessary to clone because the distorted image is created
      // by cropping, and the pixmap is thus larger than the image,
      // causing subsequent conversion to a numpy array to be misaligned. 
      im = sim->getDistorted().clone() ;
   }
   if (refLinesMode) refLines(im) ;
   return im;
}

PYBIND11_MODULE(CosmoSimPy, m) {
    m.doc() = "Wrapper for the CosmoSim simulator" ;

    m.def("helloworld", &helloworld, "A test function");

    py::class_<CosmoSim>(m, "CosmoSim")
        .def(py::init<>())
        .def("setLensMode", &CosmoSim::setLensMode)
        .def("setSourceMode", &CosmoSim::setSourceMode)
        .def("setEinsteinR", &CosmoSim::setEinsteinR)
        .def("setNterms", &CosmoSim::setNterms)
        .def("setCHI", &CosmoSim::setCHI)
        .def("setSourceParameters", &CosmoSim::setSourceParameters)
        .def("setXY", &CosmoSim::setXY)
        .def("setPolar", &CosmoSim::setPolar)
        .def("getActual", &CosmoSim::getActual)
        .def("getApparent", &CosmoSim::getSource)
        .def("getDistorted", &CosmoSim::getDistorted)
        .def("runSim", &CosmoSim::runSim)
        .def("moveSim", &CosmoSim::moveSim)
        .def("diagnostics", &CosmoSim::diagnostics)
        .def("maskImage", &CosmoSim::maskImage)
        .def("showMask", &CosmoSim::showMask)
        .def("setMaskMode", &CosmoSim::setMaskMode)
        .def("setImageSize", &CosmoSim::setImageSize)
        .def("setResolution", &CosmoSim::setResolution)
        .def("setBGColour", &CosmoSim::setBGColour)
        .def("setFile", &CosmoSim::setFile)
        ;

    pybind11::enum_<SourceSpec>(m, "SourceSpec") 
       .value( "Sphere", CSIM_SOURCE_SPHERE )
       .value( "Ellipse", CSIM_SOURCE_ELLIPSE )
       .value( "Triangle", CSIM_SOURCE_TRIANGLE ) ;
    pybind11::enum_<LensSpec>(m, "LensSpec") 
       .value( "SIS", CSIM_LENS_SPHERE )
       .value( "Ellipse", CSIM_LENS_ELLIPSE )
       .value( "PointMassRoulettes", CSIM_LENS_PM_ROULETTE ) 
       .value( "SISRoulettes", CSIM_LENS_SIS_ROULETTE ) 
       .value( "PointMass", CSIM_LENS_PM )
       .value( "Sampled", CSIM_LENS_SAMPLED )
       .value( "SampledSIS", CSIM_LENS_SAMPLED_SIS )
       .value( "PureSampled", CSIM_LENS_PURESAMPLED )
       .value( "PureSampledSIS", CSIM_LENS_PURESAMPLED_SIS )
       .value( "NoLens", CSIM_NOLENS  )  ;

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
