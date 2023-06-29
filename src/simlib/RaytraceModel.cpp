/* (C) 2023: Hans Georg Schaathun <georg@schaathun.net> */

#include "cosmosim/Simulator.h"

#include <thread>
#include "simaux.h"

RaytraceModel::RaytraceModel() :
   LensModel::LensModel()
{ 
    std::cout << "Instantiating RaytraceModel ... \n" ;
    rotatedMode = false ;
}

cv::Point2d RaytraceModel::calculateEta( cv::Point2d xi ) {
   cv::Point2d xy = cv::Point2d(
         lens->psiXvalue( xi.x, xi.y ),
         lens->psiYvalue( xi.x, xi.y ) ) ;
   /* psiXvalue/psiYvalue are defined in
    *   PsiFunctionLens, based on evaluation of analytic derivatives
    *   Lens, based on a sampled array of derivative evaluations
    * SampledPsiFunctionLens relies on the definition in Lens and calculates
    * the sampled array using a differentiation filter on a sampling of psi.
    * These differentiated arrays are used for getXi (both roulette and raytrace)
    * and for the deflection in raytrace.
    */
   return (xi - xy)/CHI ;
}
void RaytraceModel::distort(int begin, int end, const cv::Mat& src, cv::Mat& dst) {

    // std::cout << "[RaytraceModel] distort().\n" ;
    for (int row = begin; row < end; row++) {
        for (int col = 0; col < dst.cols; col++) {

            cv::Point2d eta, xi, ij, targetPos ;

            targetPos = cv::Point2d( col - dst.cols / 2.0,
                  dst.rows / 2.0 - row ) ;
            xi = -CHI*targetPos ;
            eta = calculateEta( xi ) + getEta() ;
            ij = imageCoordinate( eta, src ) ;
  
            if (ij.x < src.rows && ij.y < src.cols && ij.x >= 0 && ij.y >= 0) {
                 if ( 3 == src.channels() ) {
                    dst.at<cv::Vec3b>(row, col) = src.at<cv::Vec3b>( ij );
                 } else {
                    dst.at<uchar>(row, col) = src.at<uchar>( ij );
                 }
            }
        }
    }
}

/* getDistortedPos() is not used for raytracing, but
 * it has to be defined, since it is declared in the superclass.  */
cv::Point2d RaytraceModel::getDistortedPos(double r, double theta) const {
   throw NotImplemented() ;
};
