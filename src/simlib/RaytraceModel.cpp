/* (C) 2023: Hans Georg Schaathun <georg@schaathun.net> */

#include "cosmosim/Simulator.h"

#include <thread>
#include "simaux.h"

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
   return (xi - xy) ;
}
void RaytraceModel::distort(int begin, int end, const cv::Mat& src, cv::Mat& dst) {

    if (DEBUG) {
       std::cout << "[RaytraceModel] distort() " << lens->idString() 
          << " (" << source->idString() << ")\n" ;
    }
    for (int row = begin; row < end; row++) {
        for (int col = 0; col < dst.cols; col++) {

            cv::Point2d eta, xi, ij ;

            xi = pointCoordinate( cv::Point2d( row, col ), dst ) ;
            eta = calculateEta( xi ) - getEta() ;
            ij = imageCoordinate( eta, src ) ;
  
            if (ij.x <= src.rows-1 && ij.y <= src.cols-1 && ij.x >= 0 && ij.y >= 0) {
                 if ( 3 == src.channels() ) {
                    dst.at<cv::Vec3b>(row, col) = src.at<cv::Vec3b>( ij.x, ij.y );
                 } else {
                    dst.at<uchar>(row, col) = src.at<uchar>( ij.x, ij.y );
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

void RaytraceModel::undistort(const cv::Mat& src, cv::Mat& dst) {

    for (int row = 0; row < dst.rows; row++) {
        for (int col = 0; col < dst.cols; col++) {

            cv::Point2d eta, ij, srcPos ;

            srcPos = pointCoordinate( cv::Point2d( row, col ), src ) ;
            eta = calculateEta( srcPos ) ;
            ij = imageCoordinate( eta, dst ) ;
  
            if (ij.x <= dst.rows-1 && ij.y <= dst.cols-1 && ij.x >= 0 && ij.y >= 0) {
                 if ( 3 == src.channels() ) {
                    dst.at<cv::Vec3b>(ij.x, ij.y ) = src.at<cv::Vec3b>( row, col ) ;
                 } else {
                    int px = dst.at<uchar>(ij.x, ij.y ) = src.at<uchar>(row, col) ;
                 }
            }
        }
    }
}

/* This just splits the image space in chunks and runs distort() in parallel */
void RaytraceModel::parallelDistort(const cv::Mat& src, cv::Mat& dst) {
    int n_threads = std::thread::hardware_concurrency();

    std::vector<std::thread> threads_vec;
    int lower=0, rng=dst.rows, rng1 ; 

    rng1 = ceil( (double) rng / (double) n_threads ) ;

    for (int i = 0; i < n_threads; i++) {
        int begin = lower+rng1*i, end = begin+rng1 ;
        if ( end > dst.rows ) end = dst.rows ;
        std::thread t([begin, end, src, &dst, this]() { distort(begin, end, src, dst); });
        threads_vec.push_back(std::move(t));
    }

    for (auto& thread : threads_vec) {
        thread.join();
    }
}
