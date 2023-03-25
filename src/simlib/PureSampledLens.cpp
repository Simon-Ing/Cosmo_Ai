/* (C) 2023: Hans Georg Schaathun <georg@schaathun.net> */

#include "cosmosim/PixMap.h"

#include <thread>
#include "simaux.h"

PureSampledLens::PureSampledLens() :
   LensModel::LensModel()
{ 
    std::cout << "Instantiating PureSampledLens ... \n" ;
    rotatedMode = false ;
}
PureSampledLens::PureSampledLens(bool centred) :
   LensModel::LensModel(centred)
{ 
    std::cout << "Instantiating PureSampledLens ... \n" ;
    rotatedMode = false ;
}


void PureSampledLens::updateApparentAbs( ) {
   std::cout << "[PureSampledLens] updateApparentAbs() does nothing.\n" ;
}
cv::Point2d PureSampledLens::calculateEta( cv::Point2d xi ) {
   cv::Point2d chieta, xy, ij ; 
   cv::Mat psiX, psiY ;

   this->updatePsi() ;
   gradient( -psi, psiX, psiY ) ;
   ij = imageCoordinate( xi, psi ) ;
   xy = cv::Point2d( -psiY.at<double>( ij ), -psiX.at<double>( ij ) );
   chieta = xi - xy ;

   return chieta/CHI ;
}
void PureSampledLens::distort(int begin, int end, const cv::Mat& src, cv::Mat& dst) {

    for (int row = begin; row < end; row++) {
        for (int col = 0; col < dst.cols; col++) {

            cv::Point2d pos, ij ;

            cv::Point2d targetPos = cv::Point2d( col - dst.cols / 2.0, dst.rows / 2.0 - row ) ;
            cv::Point2d xi = CHI*targetPos ;
            cv::Point2d eta = calculateEta( xi ) ;
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
