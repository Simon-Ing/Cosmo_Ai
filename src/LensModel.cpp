/* (C) 2022: Hans Georg Schaathun <hg@schaathun.net> *
 * Building on code by Simon Ingebrigtsen, Sondre Westbø Remøy,
 * Einar Leite Austnes, and Simon Nedreberg Runde
 */

#include "Simulator.h"
#include <symengine/expression.h>
#include <symengine/lambda_double.h>
#include <thread>
#include <symengine/parser.h>
#include <fstream>

double factorial_(unsigned int n);

LensModel::LensModel() :
        CHI(0.5),
        einsteinR(20),
        nterms(10)
{ }

/* Getters for the images */
cv::Mat LensModel::getActual() { 
   cv::Mat imgApparent = getApparent() ;
   cv::Mat imgActual = cv::Mat::zeros(imgApparent.size(), imgApparent.type());

   // (x0,y0) is the centre of the image in pixel coordinates 
   double x0 = imgApparent.cols/2 ;
   double y0 = imgApparent.rows/2 ;

   cv::Point2f srcTri[3], dstTri[3];
   srcTri[0] = cv::Point2f( x0, y0 );
   dstTri[0] = cv::Point2f( x0+actualX, y0-actualY );
   srcTri[1] = cv::Point2f( x0-actualAbs, y0 );
   dstTri[1] = cv::Point2f( x0, y0 );
   srcTri[2] = cv::Point2f( x0-actualAbs, y0-actualAbs );
   dstTri[2] = cv::Point2f( x0-actualY, y0-actualX );
   cv::Mat rot = cv::getAffineTransform( srcTri, dstTri );

   std::cout << "getActual() (x,y)=(" << actualX << "," << actualY << ")\n" << rot << "\n" ;

   cv::warpAffine(imgApparent, imgActual, rot, imgApparent.size());    // crop distorted image
   return imgActual ; 
}
cv::Mat LensModel::getApparent() { return source->getImage() ; }
cv::Mat LensModel::getDistorted() { return imgDistorted ; }

cv::Mat LensModel::getSecondary() { 
   apparentAbs = apparentAbs2 ;
   this->update() ;
   return imgDistorted ; }

void LensModel::update() {

    auto startTime = std::chrono::system_clock::now();
    
    cv::Mat imgApparent = getApparent() ;

    this->calculateAlphaBeta() ;

    int nrows = imgApparent.rows ;
    int ncols = imgApparent.cols ;

    // Make Distorted Image
    // We work in a double sized image to avoid cropping
    cv::Mat imgD = cv::Mat::zeros(nrows*2, ncols*2, imgApparent.type());
    parallelDistort(imgApparent, imgD);

    // Correct the rotation applied to the source image
    double phi = atan2(actualY, actualX); // Angle relative to x-axis
    cv::Mat rot = cv::getRotationMatrix2D(cv::Point(nrows, ncols), phi*180/PI, 1);
    cv::warpAffine(imgD, imgD, rot, cv::Size(2*nrows, 2*ncols));    // crop distorted image
    imgDistorted =  imgD(cv::Rect(nrows/2, ncols/2, nrows, ncols));

    std::cout << "update() (x,y) = (" << actualX << ", " << actualY << ")\n" ;
    std::cout << rot << "\n" ;

    // Calculate run time for this function and print diagnostic output
    auto endTime = std::chrono::system_clock::now();
    std::cout << "Time to update(): " 
              << std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count() 
              << " milliseconds" << std::endl;

}


/* This just splits the image space in chunks and runs distort() in parallel */
void LensModel::parallelDistort(const cv::Mat& src, cv::Mat& dst) {
    int n_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads_vec;
    for (int i = 0; i < n_threads; i++) {
        int begin = dst.rows/n_threads*i;
        int end = dst.rows/n_threads*(i+1);
            std::thread t([begin, end, src, &dst, this]() { distort(begin, end, src, dst); });
            threads_vec.push_back(std::move(t));
        }

    for (auto& thread : threads_vec) {
        thread.join();
    }
}


void LensModel::distort(int begin, int end, const cv::Mat& src, cv::Mat& dst) {
    // Iterate over the pixels in the image distorted image.
    // (row,col) are pixel co-ordinates
    for (int row = begin; row < end; row++) {
        for (int col = 0; col < dst.cols; col++) {

            int row_, col_;  // pixel co-ordinates in the apparent image
            std::pair<double, double> pos ;

            // Set coordinate system with origin at the centre of mass
            // in the distorted image in the lens plane.
            double x = (col - dst.cols / 2.0 - apparentAbs) * CHI;
            double y = (dst.rows / 2.0 - row) * CHI;
            // (x,y) are coordinates in the lens plane, and hence the
            // multiplication by CHI

            // Calculate distance and angle of the point evaluated 
            // relative to center of lens (origin)
            double r = sqrt(x * x + y * y);
            double theta = x == 0 ? 0 : atan2(y, x);

            pos = this->getDistortedPos(r, theta);

            // Translate to array index in the source plane
            row_ = (int) round(src.rows / 2.0 - pos.second);
            col_ = (int) round(src.cols / 2.0 + pos.first);

            // If (x', y') within source, copy value to imgDistorted
            if (row_ < src.rows && col_ < src.cols && row_ >= 0 && col_ >= 0) {
               if ( 3 == src.channels() ) {
                  dst.at<cv::Vec3b>(row, col) = src.at<cv::Vec3b>(row_, col_);
               } else {
                  dst.at<uchar>(row, col) = src.at<uchar>(row_, col_);
               }
            }
        }
    }
}

/* Calculate n! (n factorial) */
double factorial_(unsigned int n){
    double a = 1.0;
    for (int i = 2; i <= n; i++){
        a *= i;
    }
    return a;
}

void LensModel::updateNterms(int n) {
   nterms = n ;
   update() ;
}
void LensModel::updateAll( double X, double Y, double er, double chi, int n) {
   nterms = n ;
   updateXY(X,Y,chi,er);
}

/* Re-calculate co-ordinates using updated parameter settings from the GUI.
 * This is called from the update() method.                                  */
void LensModel::updateXY( double X, double Y, double chi, double er ) {

    CHI = chi ;
    einsteinR = er ;
    // Actual position in source plane
    actualX = X ;
    actualY = Y ;

    // Absolute values in source plane
    actualAbs = sqrt(actualX * actualX + actualY * actualY); // Actual distance from the origin

    // The apparent position is the solution to a quadratic equation.
    // thus there are two solutions.
    double root = sqrt(0.25*actualAbs*actualAbs + einsteinR*einsteinR/(CHI*CHI));
    apparentAbs = actualAbs/2 + root ;
    apparentAbs2 = actualAbs/2 - root ;
    // BDN: Is the calculation of apparent positions correct above?

    update() ;
}

void LensModel::setSource(Source *src) {
    source = src ;
}

/* Default implementation doing nothing.
 * This is correct for any subclass that does not need the alpha/beta tables. */
void LensModel::calculateAlphaBeta() { }

