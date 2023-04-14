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

double CosmoSim::getAlphaXi( int m, int s ) {
      if ( NULL == lens ) throw NotSupported();
      return lens->getAlphaXi( m, s ) ;
}
double CosmoSim::getBetaXi( int m, int s ) {
      if ( NULL == lens ) throw NotSupported();
      return lens->getBetaXi( m, s ) ;
}
double CosmoSim::getAlpha(
      cv::Point2d xi, int m, int s 
 ) {
      if ( NULL == lens ) throw NotSupported();
      return lens->getAlpha( xi, m, s ) ;
}
double CosmoSim::getBeta( 
      cv::Point2d xi, int m, int s 
) {
      if ( NULL == lens ) throw NotSupported();
      return lens->getBeta( xi, m, s ) ;
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

cv::Mat CosmoSim::getPsiMap( ) {
   cv::Mat im = lens->getPsi() ;
   std::cout << "[getPsiMap] " << im.type() << "\n" ;
   return im ;
} 
cv::Mat CosmoSim::getMassMap( ) {
   cv::Mat im = lens->getMassMap() ;
   std::cout << "[getMassMap] " << im.type() << "\n" ;
   return im ;
} 

#define makeSIS(sim)  lens = new SIS() ; lens->setFile(filename) ; sim->setLens(lens) 
void CosmoSim::setCHI(double c) { chi = c/100.0 ; }
void CosmoSim::setNterms(int c) { nterms = c ; }
void CosmoSim::setXY( double x, double y) { xPos = x ; yPos = y ; rPos = -1 ; }
void CosmoSim::setPolar(int r, int theta) { rPos = r ; thetaPos = theta ; }
void CosmoSim::setLensMode(int m) { lensmode = m ; }
void CosmoSim::setLensFunction(int m) { psimode = m ; }
void CosmoSim::setSourceMode(int m) { srcmode = m ; }
void CosmoSim::setMaskMode(bool b) { maskmode = b ; }
void CosmoSim::setBGColour(int b) { bgcolour = b ; }
void CosmoSim::initLens() {
   PsiFunctionModel *s1 = NULL ;
   PsiFunctionLens *l1 = NULL ;
   bool centred = false ;
   std::cout << "[CosmoSim.cpp] initLens\n" ;
   if ( lensmode == oldlensmode ) return ;
   if ( sim ) delete sim ;
   switch ( psimode ) {
       case CSIM_PSI_SIS:
          lens = new SIS() ;
          break ;
       case CSIM_NOPSI:
          lens = NULL ;
          break ;
       default:
         std::cout << "No such lens model!\n" ;
         throw NotImplemented();
   }
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
         std::cout << "Running Roulette Point Mass Lens (mode=" 
                   << lensmode << ")\n" ;
         sim = new RoulettePMLens(centred) ;
         break ;
       case CSIM_LENS_PM:
         std::cout << "Running Point Mass Lens (mode=" << lensmode << ")\n" ;
         sim = new PointMassLens(centred) ;
         break ;
       case CSIM_LENS_PURESAMPLED_SIS:
         std::cout << "Running Pure Sampled SIS Lens (mode=" << lensmode << ")\n" ;
         sim = new PureSampledModel(centred) ;
         makeSIS( sim ) ;
         break ;
       case CSIM_LENS_PSIFUNCTION_SIS:
         std::cout << "Running Pure Sampled SIS Lens (mode=" << lensmode << ")\n" ;
         s1 = new PsiFunctionModel(centred) ;
         s1->setPsiFunctionLens( l1 = new SIS() ) ;
         lens = l1 ;
         sim = s1 ;
         break ;
       case CSIM_LENS_SAMPLED_SIS:
         std::cout << "Running Sampled SIS Lens (mode=" << lensmode << ")\n" ;
         sim = new SampledRouletteLens(centred) ;
         makeSIS( sim ) ;
         break ;
       case CSIM_LENS_ROULETTE_SIS:
         std::cout << "Running Sampled SIS Lens (mode=" << lensmode << ")\n" ;
         sim = new RouletteLens(centred) ;
         makeSIS( sim ) ;
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
   if ( lens != NULL ) {
      lens->setEinsteinR( einsteinR ) ;
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
        .def("setLensFunction", &CosmoSim::setLensFunction)
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
        .def("getPsiMap", &CosmoSim::getPsiMap)
        .def("getMassMap", &CosmoSim::getMassMap)
        .def("getAlpha", &CosmoSim::getAlpha)
        .def("getBeta", &CosmoSim::getBeta)
        .def("getAlphaXi", &CosmoSim::getAlphaXi)
        .def("getBetaXi", &CosmoSim::getBetaXi)
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
       .value( "SampledSIS", CSIM_LENS_SAMPLED_SIS )
       .value( "PureSampledSIS", CSIM_LENS_PURESAMPLED_SIS )
       .value( "PsiFunctionSIS", CSIM_LENS_PSIFUNCTION_SIS )
       .value( "RouletteSIS", CSIM_LENS_ROULETTE_SIS )
       .value( "NoLens", CSIM_NOLENS  )  ;

    // cv::Mat binding from https://alexsm.com/pybind11-buffer-protocol-opencv-to-numpy/
    pybind11::class_<cv::Mat>(m, "Image", pybind11::buffer_protocol())
        .def_buffer([](cv::Mat& im) -> pybind11::buffer_info {
              int t = im.type() ;
              if ( (t&CV_64F) == CV_64F ) {
                std::cout << "[CosmoSimPy] CV_64F\n" ;
                return pybind11::buffer_info(
                    // Pointer to buffer
                    im.data,
                    // Size of one scalar
                    sizeof(double),
                    // Python struct-style format descriptor
                    pybind11::format_descriptor<double>::format(),
                    // Number of dimensions
                    3,
                        // Buffer dimensions
                    { im.rows, im.cols, im.channels() },
                    // Strides (in bytes) for each index
                    {
                        sizeof(double) * im.channels() * im.cols,
                        sizeof(double) * im.channels(),
                        sizeof(unsigned char)
                    }
                    );
              } else { // default is 8bit integer
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
              } ;
        });
    // Note.  The cv::Mat object returned needs to by wrapped in python:
    // `np.array(im, copy=False)` where `im` is the `Mat` object.

}
