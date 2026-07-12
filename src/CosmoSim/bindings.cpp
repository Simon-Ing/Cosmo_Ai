/* (C) 2026: Hans Georg Schaathun <georg@schaathun.net> */

#include "cosmosim/Roulette.h"
#include "cosmosim/Simulator.h"
#include "cosmosim/Source.h"
#include "cosmosim/Lens.h"

#include <opencv2/opencv.hpp>

#include <pybind11/pybind11.h>

#ifndef DEBUG
#define DEBUG debug
#endif

enum SourceSpec { CSIM_SOURCE_SPHERE,
                  CSIM_SOURCE_ELLIPSE,
                  CSIM_SOURCE_IMAGE,
                  CSIM_SOURCE_TRIANGLE,
                  CSIM_SOURCE_EXTERN } ;
enum ModelSpec { CSIM_MODEL_RAYTRACE,
                  CSIM_MODEL_ROULETTE,
                  CSIM_MODEL_ROULETTE_REGEN,
                  CSIM_NOMODEL } ;
enum PsiSpec    { CSIM_PSI_SIS,
                  CSIM_PSI_SIE,
                  CSIM_PSI_CLUSTER,
                  CSIM_PSI_PM,
                  CSIM_NOPSI } ;
namespace py = pybind11;

PYBIND11_MODULE(CosmoSimPy, m) {
    m.doc() = "Wrapper for the CosmoSim simulator" ;

    m.def("getDebug", []() { return debug ; });
    m.def("setDebug", [](int val) { 
	  if (DEBUG) std::cout << "[setDebug] " << val << std::endl ;
	  debug  = val; });

    py::class_<Source>(m, "Source")
        .def(py::init<int>())
        .def("getImage", &Source::getImage)
        ;
    py::class_<SphericalSource,Source>(m, "SphericalSource")
        .def(py::init<int,double,double,double,LightProfileSpec>())
        ;
    py::class_<EllipsoidSource,Source>(m, "EllipsoidSource")
        .def(py::init<int,double,double,double,LightProfileSpec>())
        .def(py::init<int,double,double,double,LightProfileSpec,double,double>())
        ;
    py::class_<SourceConstellation,Source>(m, "SourceConstellation")
        .def(py::init<int>())
        .def("addSource", &SourceConstellation::addSource)
        ;
    py::class_<TriangleSource,Source>(m, "TriangleSource")
        .def(py::init<int,double,double>())
        ;
    py::class_<ImageSource,Source>(m, "ImageSource")
        .def(py::init<int,std::string>())
        ;
    py::class_<Lens>(m, "Lens")
        .def(py::init<>())
        .def("calculateAlphaBeta", &Lens::calculateAlphaBeta)
        .def("getAlphaXi", &Lens::getAlphaXi)
        .def("getBetaXi", &Lens::getBetaXi)
        .def("getXi", &Lens::getXi)
        .def("psiValue", &Lens::psiValue)
        .def("psiXvalue", &Lens::psiXvalue)
        .def("psiYvalue", &Lens::psiYvalue)
        .def("criticalXi", &Lens::criticalXi)
        .def("caustic", &Lens::caustic)
        .def("getAlpha", &Lens::getAlphaPy)
        .def("getBeta", &Lens::getBetaPy)
        ;
    py::class_<SampledLens,Lens>(m, "SampledLens") ;
    py::class_<SampledPsiFunctionLens,SampledLens,Lens>(m, "SampledPsiFunctionLens") 
        .def(py::init<PsiFunctionLens *>())
        .def(py::init<PsiFunctionLens *,int>())
        .def("criticalXi", &SampledPsiFunctionLens::criticalXi) ;
    py::class_<PsiFunctionLens,Lens>(m, "PsiFunctionLens")
        .def(py::init<>())
        .def("calculateAlphaBeta", &PsiFunctionLens::calculateAlphaBeta)
        .def("getAlphaXi", &PsiFunctionLens::getAlphaXi)
        .def("getBetaXi", &PsiFunctionLens::getBetaXi)
        .def("getAlpha", &PsiFunctionLens::getAlpha)
        .def("getBeta", &PsiFunctionLens::getBeta)
        .def("setEinsteinR", &PsiFunctionLens::setEinsteinR)
        .def("setOrientation", &PsiFunctionLens::setOrientation)
        .def("setRatio", &PsiFunctionLens::setRatio)
        .def("setFile", &PsiFunctionLens::setFile)
        .def("initAlphasBetas", &PsiFunctionLens::initAlphasBetas)
        .def("getEinsteinR", &PsiFunctionLens::getEinsteinR)
        ;
    py::class_<SIS,PsiFunctionLens>(m, "SIS")
        .def(py::init<>())
        .def("psiValue", &SIS::psiValue)
        .def("psiXvalue", &SIS::psiXvalue)
        .def("psiYvalue", &SIS::psiYvalue)
        .def("getXi", &SIS::getXi)
        .def("criticalXi", &SIS::criticalXi)
        ;
    py::class_<SIE,PsiFunctionLens>(m, "SIE")
        .def(py::init<>())
        .def("psiValue", &SIE::psiValue)
        .def("psiXvalue", &SIE::psiXvalue)
        .def("psiYvalue", &SIE::psiYvalue)
        .def("getXi", &SIE::getXi)
        .def("criticalXi", &SIE::criticalXi)
        ;
    py::class_<PointMass,PsiFunctionLens>(m, "PointMass")
        .def(py::init<>())
        .def("psiValue", &PointMass::psiValue)
        .def("psiXvalue", &PointMass::psiXvalue)
        .def("psiYvalue", &PointMass::psiYvalue)
        .def("getXi", &PointMass::getXi)
        ;
    py::class_<ClusterLens,PsiFunctionLens>(m, "ClusterLens")
        .def(py::init<>())
        .def("addLens", &ClusterLens::addLens)
        .def("calculateAlphaBeta", &ClusterLens::calculateAlphaBeta)
        .def("psiValue", &ClusterLens::psiValue)
        .def("psiXvalue", &ClusterLens::psiXvalue)
        .def("psiYvalue", &ClusterLens::psiYvalue)
        .def("initAlphasBetas", &ClusterLens::initAlphasBetas)
        ;

    py::class_<SimulatorModel>(m, "SimulatorModel")
        .def(py::init<>())
        .def("update", py::overload_cast<>(&SimulatorModel::update))
        .def("setSource", &SimulatorModel::setSource)
        .def("setXY", &SimulatorModel::setXY)
        .def("setPolar", &SimulatorModel::setPolar)
        .def("setNterms", &SimulatorModel::setNterms)
        .def("setMaskMode", &SimulatorModel::setMaskMode)
        .def("setBGColour", &SimulatorModel::setBGColour)
        .def("setMaskRadius", &SimulatorModel::setMaskRadius)
        .def("getActual", &SimulatorModel::getActual)
        .def("getDistorted", &SimulatorModel::getDistorted) 
        .def("getRelativeEta", &SimulatorModel::getRelativeEtaPy)
        .def("getOffset", &SimulatorModel::getOffset)
        .def("getEta", &SimulatorModel::getEta)
        .def("getNu", &SimulatorModel::getNu)
        .def("getCritical", &SimulatorModel::getCritical) 
        .def("drawCritical", &SimulatorModel::drawCritical) 
        .def("drawCaustics", &SimulatorModel::drawCaustics) 
        .def("setLens", &SimulatorModel::setLens) ;
    py::class_<RouletteModel,SimulatorModel>(m, "RouletteModel")
        .def(py::init<>())
        .def("setLens", &RouletteModel::setLens) ;
    py::class_<RouletteRegenerator,RouletteModel>(m, "RouletteRegenerator")
        .def(py::init<>())
        .def("setCentrePy", &RouletteRegenerator::setCentrePy)
        .def("setAlphaXi", &RouletteRegenerator::setAlphaXi)
        .def("setBetaXi", &RouletteRegenerator::setBetaXi) ;

    py::class_<RaytraceModel,SimulatorModel>(m, "RaytraceModel")
        .def(py::init<>()) ;

    pybind11::enum_<PsiSpec>(m, "PsiSpec") 
       .value( "SIE", CSIM_PSI_SIE )
       .value( "SIS", CSIM_PSI_SIS )
       .value( "Cluster", CSIM_PSI_CLUSTER )
       .value( "PM", CSIM_PSI_PM ) 
       .value( "NoPsi", CSIM_NOPSI ) ;
    pybind11::enum_<SourceSpec>(m, "SourceSpec") 
       .value( "Sphere", CSIM_SOURCE_SPHERE )
       .value( "Ellipse", CSIM_SOURCE_ELLIPSE )
       .value( "Image", CSIM_SOURCE_IMAGE )
       .value( "Triangle", CSIM_SOURCE_TRIANGLE ) ;
    pybind11::enum_<ModelSpec>(m, "ModelSpec") 
       .value( "Raytrace", CSIM_MODEL_RAYTRACE )
       .value( "Roulette", CSIM_MODEL_ROULETTE  )
       .value( "RouletteRegenerator", CSIM_MODEL_ROULETTE_REGEN  )
       .value( "NoModel", CSIM_NOMODEL  )  ;
    pybind11::enum_<LightProfileSpec>(m, "LightProfileSpec") 
       .value( "Gaussian", CSIM_LIGHT_GAUSSIAN )
       .value( "Sersic", CSIM_LIGHT_SERSIC );

    // cv::Mat binding from https://alexsm.com/pybind11-buffer-protocol-opencv-to-numpy/
    pybind11::class_<cv::Mat>(m, "Image", pybind11::buffer_protocol())
        .def_buffer([](cv::Mat& im) -> pybind11::buffer_info {
              int t = im.type() ;
              if ( (t&CV_64F) == CV_64F ) {
                if (DEBUG) std::cout << "[CosmoSimPy] CV_64F\n" ;
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
        .def(py::init<>())
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
