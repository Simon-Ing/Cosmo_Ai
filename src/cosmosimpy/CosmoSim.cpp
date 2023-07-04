/* (C) 2022-23: Hans Georg Schaathun <georg@schaathun.net> */

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

double CosmoSim::getChi( ) { return chi ; } ;
cv::Point2d CosmoSim::getRelativeEta( double x, double y ) {
   // Input (x,y) is the centre point $\nu$
   cv::Point2d pt = sim->getRelativeEta( cv::Point2d( x,y )*chi ) ; 
   std::cout << "[CosmoSim::getRelativeEta] " << pt << "\n" ;
   return pt ;
} ;
cv::Point2d CosmoSim::getOffset( double x, double y ) {
   // Input (x,y) is the centre point $\nu$ 
   cv::Point2d pt = sim->getOffset( cv::Point2d( x,y )*chi ) ; 
   std::cout << "[CosmoSim::getOffset] " << pt << "\n" ;
   return pt ;
} ;

double CosmoSim::getAlphaXi( int m, int s ) {
      if ( NULL != psilens )
          return psilens->getAlphaXi( m, s ) ;
      else if ( NULL != lens )
          return lens->getAlphaXi( m, s ) ;
      else throw NotSupported();
}
double CosmoSim::getBetaXi( int m, int s ) {
      if ( NULL != psilens )
          return psilens->getBetaXi( m, s ) ;
      else if ( NULL != lens )
          return lens->getBetaXi( m, s ) ;
      else throw NotSupported();
}
double CosmoSim::getAlpha(
      double x, double y, int m, int s 
 ) {
      double r ;
      cv::Point2d xi = cv::Point2d( x, y )*chi ;
      if ( NULL != psilens )
          r = psilens->getAlpha( xi, m, s ) ;
      else if ( NULL != lens )
          r = lens->getAlpha( xi, m, s ) ;
      else throw NotSupported();
      // std::cout << "getAlpha(" << xi << ") " << r << "\n" ;
      return r ;
}
double CosmoSim::getBeta( 
      double x, double y, int m, int s 
) {
      double r ;
      cv::Point2d xi = cv::Point2d( x, y )*chi ;
      if ( NULL != psilens )
          r = psilens->getBeta( xi, m, s ) ;
      else if ( NULL != lens )
          r = lens->getBeta( xi, m, s ) ;
      else throw NotSupported();
      // std::cout << "getBeta(" << xi << ") " << r << "\n" ;
      return r ;
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

void CosmoSim::setCHI(double c) { chi = c/100.0 ; }
void CosmoSim::setNterms(int c) { nterms = c ; }
void CosmoSim::setXY( double x, double y) { xPos = x ; yPos = y ; rPos = -1 ; }
void CosmoSim::setPolar(int r, int theta) { rPos = r ; thetaPos = theta ; }
void CosmoSim::setModelMode(int m) { 
   if ( modelmode != m ) {
      modelmode = m ; 
      modelchanged = 1 ;
   }
}
void CosmoSim::setLensMode(int m) { 
   if ( lensmode != m ) {
      std::cout << "[CosmoSim.cpp] setLensMode(" << lensmode 
         << " -> " << m << ")\n" ;
      lensmode = m ; 
      modelchanged = 1 ;
   } else {
      std::cout << "[CosmoSim.cpp] setLensMode(" << lensmode << ") unchanged\n" ;
   }
}
void CosmoSim::setSampled(int m) { 
   if ( sampledlens != m ) {
      sampledlens = m ; 
      modelchanged = 1 ;
   }
}
void CosmoSim::setSourceMode(int m) { srcmode = m ; }
void CosmoSim::setMaskMode(bool b) { maskmode = b ; }
void CosmoSim::setBGColour(int b) { bgcolour = b ; }
void CosmoSim::initLens() {
   std::cout << "[CosmoSim.cpp] initLens\n" ;
   if ( ! modelchanged ) return ;
   if ( sim ) delete sim ;
   psilens = NULL ;
   switch ( lensmode ) {
       case CSIM_PSI_SIS:
          std::cout << "[initLens] SIS\n" ;
          lens = psilens = new SIS() ;
          lens->setFile(filename) ;
          lens->initAlphasBetas() ;
          break ;
       case CSIM_NOPSI_PM:
       case CSIM_NOPSI_SIS:
       case CSIM_NOPSI:
          std::cout << "[initLens] Point Mass or No Lens (" 
                << lensmode << ")\n" ;
          lens = NULL ;
          break ;
       default:
         std::cout << "No such lens model!\n" ;
         throw NotImplemented();
   }
   if ( sampledlens ) {
     lens = new SampledPsiFunctionLens( psilens ) ;
     lens->setFile(filename) ;
   }
   switch ( modelmode ) {
       case CSIM_MODEL_SIS_ROULETTE:
         std::cout << "Running SphereLens (mode=" << modelmode << ")\n" ;
         sim = new SphereLens(filename) ;
         break ;
       case CSIM_MODEL_POINTMASS_ROULETTE:
         std::cout << "Running Roulette Point Mass Lens (mode=" 
                   << modelmode << ")\n" ;
         sim = new RoulettePMLens() ;
         break ;
       case CSIM_MODEL_POINTMASS_EXACT:
         std::cout << "Running Point Mass Lens (mode=" << modelmode << ")\n" ;
         sim = new PointMassLens() ;
         break ;
       case CSIM_MODEL_RAYTRACE:
         std::cout << "Running Raytrace Lens (mode=" << modelmode << ")\n" ;
         sim = new RaytraceModel() ;
         sim->setLens(lens) ;
         break ;
       case CSIM_MODEL_ROULETTE:
         std::cout << "Running Roulette Lens (mode=" << modelmode << ")\n" ;
         sim = new RouletteModel() ;
         sim->setLens(lens) ;
         break ;
       case CSIM_NOMODEL:
         std::cout << "Specified No Model.\n" ;
         throw NotImplemented();
       default:
         std::cout << "No such lens mode!\n" ;
         throw NotImplemented();
    }
    modelchanged = 0 ;
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
   if ( running ) {
      std::cout << "[CosmoSim.cpp] runSim() - simulator already running.\n" ;
      return false ;
   }
   std::cout << "[CosmoSim.cpp] runSim() - running similator\n" ;
   initLens() ;
   initSource() ;
   sim->setBGColour( bgcolour ) ;
   sim->setNterms( nterms ) ;
   if ( lens != NULL ) lens->setNterms( nterms ) ;
   sim->setMaskMode( maskmode ) ;
   if ( CSIM_NOPSI_ROULETTE != lensmode ) {
      if ( rPos < 0 ) {
         sim->setXY( xPos, yPos, chi, einsteinR ) ;
      } else {
         sim->setPolar( rPos, thetaPos, chi, einsteinR ) ;
      }
      if ( lens != NULL ) {
         lens->setEinsteinR( einsteinR ) ;
      }
   }
   std::cout << "[runSim] set parameters, ready to run\n" ;
   Py_BEGIN_ALLOW_THREADS
   std::cout << "[runSim] thread section\n" ;
   if ( sim == NULL )
      throw std::logic_error("Simulator not initialised") ;
   sim->update() ;
   std::cout << "[CosmoSim.cpp] end of thread section\n" ;
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
   if ( sim == NULL )
      throw std::logic_error("Simulator not initialised") ;
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
   im = sim->getDistorted() ;
   std::cout << "[getDistorted] size=" << im.size << "\n" ;
   if ( basesize < size ) {
      std::cout << "basesize=" << basesize << "; size=" << size << "\n" ;
      cv::Mat ret(cv::Size(basesize, basesize), sim->getActual().type(),
                  cv::Scalar::all(255));
      cv::resize(im,ret,cv::Size(basesize,basesize) ) ;
      im = ret ;
   } else {
      // It is necessary to clone because the distorted image is created
      // by cropping, and the pixmap is thus larger than the image,
      // causing subsequent conversion to a numpy array to be misaligned. 
      im = im.clone() ;
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
        .def("setModelMode", &CosmoSim::setModelMode)
        .def("setSampled", &CosmoSim::setSampled)
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
        .def("getChi", &CosmoSim::getChi)
        .def("getOffset", &CosmoSim::getOffset)
        .def("getRelativeEta", &CosmoSim::getRelativeEta)
        ;

    py::class_<RouletteSim>(m, "RouletteSim")
        .def(py::init<>())
        .def("setSourceMode", &RouletteSim::setSourceMode)
        .def("setNterms", &RouletteSim::setNterms)
        .def("setSourceParameters", &RouletteSim::setSourceParameters)
        .def("getActual", &RouletteSim::getActual)
        .def("getApparent", &RouletteSim::getSource)
        .def("getDistorted", &RouletteSim::getDistorted)
        .def("runSim", &RouletteSim::runSim)
        .def("initSim", &RouletteSim::initSim)
        .def("diagnostics", &RouletteSim::diagnostics)
        .def("maskImage", &RouletteSim::maskImage)
        .def("showMask", &RouletteSim::showMask)
        .def("setMaskMode", &RouletteSim::setMaskMode)
        .def("setImageSize", &RouletteSim::setImageSize)
        .def("setResolution", &RouletteSim::setResolution)
        .def("setBGColour", &RouletteSim::setBGColour)
        .def("setAlphaXi", &RouletteSim::setAlphaXi)
        .def("setBetaXi", &RouletteSim::setBetaXi) ;

    pybind11::enum_<PsiSpec>(m, "PsiSpec") 
       .value( "SIS", CSIM_PSI_SIS )
       .value( "PM", CSIM_NOPSI_PM ) 
       .value( "Roulette", CSIM_NOPSI_ROULETTE ) 
       .value( "NoPsiSIS", CSIM_NOPSI_SIS ) 
       .value( "NoPsi", CSIM_NOPSI ) ;
    pybind11::enum_<SourceSpec>(m, "SourceSpec") 
       .value( "Sphere", CSIM_SOURCE_SPHERE )
       .value( "Ellipse", CSIM_SOURCE_ELLIPSE )
       .value( "Triangle", CSIM_SOURCE_TRIANGLE ) ;
    pybind11::enum_<ModelSpec>(m, "ModelSpec") 
       .value( "Raytrace", CSIM_MODEL_RAYTRACE )
       .value( "Roulette", CSIM_MODEL_ROULETTE  )
       .value( "RouletteRegenerator", CSIM_MODEL_ROULETTE_REGEN  )
       .value( "PointMassExact", CSIM_MODEL_POINTMASS_EXACT )
       .value( "PointMassRoulettes", CSIM_MODEL_POINTMASS_ROULETTE ) 
       .value( "SIS", CSIM_MODEL_SIS_ROULETTE  )
       .value( "NoModel", CSIM_NOMODEL  )  ;

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

    pybind11::class_<cv::Point2d>(m, "Point", pybind11::buffer_protocol())
        .def_buffer([](cv::Point2d& pt) -> pybind11::buffer_info {
                return pybind11::buffer_info(
                    // Pointer to buffer
                    & pt,
                    // Size of one scalar
                    sizeof(double),
                    // Python struct-style format descriptor
                    pybind11::format_descriptor<double>::format(),
                    // Number of dimensions
                    1,
                        // Buffer dimensions
                    { 2 },
                    // Strides (in bytes) for each index
                    {
                        sizeof(double)
                    }
                 );
        });

}
