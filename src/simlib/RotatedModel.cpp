/* (C) 2023: Hans Georg Schaathun <georg@schaathun.net> *
 *
 * RotatedModel assumes that the source is located on the $x$ axis.
 * Thus we override three functions.
 * 1.  getApparent() is overridden to rotate the source to fall on the $x$ axis.
 * 2.  updateInner() is overridden to apply the inverse rotation on the distorted image.
 * 3.  updateApparentAbs() is overridden so that xi and nu are rotated so that y=0. 
 */

#include "cosmosim/Simulator.h"
#include "simaux.h"

cv::Mat RotatedModel::getApparent() const {
   cv::Mat src, dst ;
   src = source->getImage() ;
   int nrows = src.rows ;
   int ncols = src.cols ;
   cv::Mat rot = cv::getRotationMatrix2D(cv::Point(nrows/2, ncols/2),
             360-phi*180/PI, 1) ;
   cv::warpAffine(src, dst, rot, src.size() ) ;
   std::cout << "[RotatedModel] getApparent()\n" ;
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

void RotatedModel::updateApparentAbs( ) {
    std::cout << "[LensModel] updateApparentAbs() updates psi.\n" ;
    cv::Mat im = getActual() ;
    lens->updatePsi(im.size()) ;
    cv::Point2d chieta = cv::Point2d( CHI*getEtaAbs(), 0 ) ;
    cv::Point2d xi1 = lens->getXi( chieta ) ;
    setNu( xi1/CHI ) ;
}
