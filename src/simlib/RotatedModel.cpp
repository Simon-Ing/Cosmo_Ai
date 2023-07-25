/* (C) 2023: Hans Georg Schaathun <georg@schaathun.net> *
 * Building on code by Simon Ingebrigtsen, Sondre Westbø Remøy,
 * Einar Leite Austnes, and Simon Nedreberg Runde
 */

#include "cosmosim/Simulator.h"
#include "simaux.h"

#include <thread>

#define DEBUG 0

double factorial_(unsigned int n);

#define signf(y)  ( y < 0 ? -1 : +1 )

cv::Mat RotatedModel::getApparent() const {
   cv::Mat src, dst ;
   src = source->getImage() ;
   int nrows = src.rows ;
   int ncols = src.cols ;
   cv::Mat rot = cv::getRotationMatrix2D(cv::Point(nrows/2, ncols/2),
             360-phi*180/PI, 1) ;
   cv::warpAffine(src, dst, rot, src.size() ) ;
   return dst ;
}

void RotatedModel::updateInner( ) {
    cv::Mat imgApparent = getApparent() ;

    std::cout << "[RotatedModel::updateInner()] R=" << getEtaAbs() << "; theta=" << phi
              << "; CHI=" << CHI << "\n" ;
    std::cout << "[RotatedModel::updateInner()] xi=" << getXi()   
              << "; eta=" << getEta() << "; etaOffset=" << etaOffset << "\n" ;
    std::cout << "[RotatedModel::updateInner()] nu=" << getNu()   
              << "; centre=" << getCentre() << "\n" << std::flush ;

    auto startTime = std::chrono::system_clock::now();

    int nrows = imgApparent.rows ;
    int ncols = imgApparent.cols ;

    // Make Distorted Image
    // We work in a double sized image to avoid cropping
    cv::Mat imgD = cv::Mat(nrows*2, ncols*2, imgApparent.type(), 0.0 ) ;
    parallelDistort(imgApparent, imgD);
  
    // Correct the rotation applied to the source image
    cv::Mat rot = cv::getRotationMatrix2D(cv::Point(nrows, ncols), phi*180/PI, 1);
    cv::warpAffine(imgD, imgD, rot, cv::Size(2*nrows, 2*ncols));    
    // crop distorted image
    imgDistorted = imgD(cv::Rect(nrows/2, ncols/2, nrows, ncols)) ;

    // Calculate run time for this function and print diagnostic output
    auto endTime = std::chrono::system_clock::now();
    std::cout << "Time to update(): " 
              << std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count() 
              << " milliseconds" << std::endl << std::flush ;

}

