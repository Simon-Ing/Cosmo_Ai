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

Simulator::Simulator(int s) :
        size(s),
        CHI(0.5),
        einsteinR(size/20),
        nterms(10)
{ }
Simulator::Simulator() : Simulator(500) {};

/* Getters for the images */
cv::Mat Simulator::getActual() { 
   cv::Mat imgApparent = getApparent() ;
   cv::Mat imgActual(size, size, CV_8UC1, cv::Scalar(0, 0, 0));
   double phi = atan2(actualY, actualX); // Angle relative to x-axis
   cv::Mat rot = cv::getRotationMatrix2D(cv::Point(size, size), phi*180/PI, 1);
   rot.at<uchar>(0,2) = actualX ;
   rot.at<uchar>(1,2) = actualY ;
   cv::warpAffine(imgApparent, imgActual, rot, cv::Size(size, size));    // crop distorted image
   return imgActual ; 
}
cv::Mat Simulator::getApparent() { return source->getImage() ; }
cv::Mat Simulator::getDistorted() { return imgDistorted ; }

cv::Mat Simulator::getSecondary() { 
   apparentAbs = apparentAbs2 ;
   this->update() ;
   return imgDistorted ; }

void Simulator::update() {

    auto startTime = std::chrono::system_clock::now();
    
    cv::Mat imgApparent = getApparent() ;

    this->calculateAlphaBeta() ;

    // Make Distorted Image
    // We work in a double sized image to avoid cropping
    cv::Mat imgD = cv::Mat(size*2, size*2, CV_8UC1, cv::Scalar(0, 0, 0));
    parallelDistort(imgApparent, imgD);

    // Correct the rotation applied to the source image
    double phi = atan2(actualY, actualX); // Angle relative to x-axis
    cv::Mat rot = cv::getRotationMatrix2D(cv::Point(size, size), phi*180/PI, 1);
    cv::warpAffine(imgD, imgD, rot, cv::Size(2*size, 2*size));    // crop distorted image
    imgDistorted =  imgD(cv::Rect(size/2, size/2, size, size));

    // Calculate run time for this function and print diagnostic output
    auto endTime = std::chrono::system_clock::now();
    std::cout << "Time to update(): " 
              << std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count() 
              << " milliseconds" << std::endl;

}


/* This just splits the image space in chunks and runs distort() in parallel */
void Simulator::parallelDistort(const cv::Mat& src, cv::Mat& dst) {
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


void Simulator::distort(int begin, int end, const cv::Mat& src, cv::Mat& dst) {
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
                auto val = src.at<uchar>(row_, col_);
                dst.at<uchar>(row, col) = val;
            }
        }
    }
}


/* The following is a default implementation for the point mass lens. 
 * It would be better to make the class abstract and move this definition to the 
 * subclass. */
std::pair<double, double> Simulator::getDistortedPos(double r, double theta) const {
    double R = apparentAbs * CHI ;
    double frac = (einsteinR * einsteinR * r) / (r * r + R * R + 2 * r * R * cos(theta));
    double nu1 = r*cos(theta) + frac * (r / R + cos(theta)) ;
    double nu2 = r*sin(theta) - frac * sin(theta) ;
    return { nu1/CHI, nu2/CHI };
}


/* Calculate n! (n factorial) */
double factorial_(unsigned int n){
    double a = 1.0;
    for (int i = 2; i <= n; i++){
        a *= i;
    }
    return a;
}

void Simulator::updateNterms(int n) {
   nterms = n ;
   update() ;
}
void Simulator::updateAll( double X, double Y, double er, double chi, int n) {
   nterms = n ;
   updateXY(X,Y,chi,er);
}

/* Re-calculate co-ordinates using updated parameter settings from the GUI.
 * This is called from the update() method.                                  */
void Simulator::updateXY( double X, double Y, double chi, double er ) {

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

void Simulator::setSource(Source *src) {
    source = src ;
}

/* Default implementation doing nothing.
 * This is correct for any subclass that does not need the alpha/beta tables. */
void Simulator::calculateAlphaBeta() { }
