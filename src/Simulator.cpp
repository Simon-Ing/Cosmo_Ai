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
        sourceSize(size/20),
        nterms(10)
{ 
    imgActual = cv::Mat(size, size, CV_8UC1, cv::Scalar(0, 0, 0));
    imgApparent = cv::Mat(size, 2*size, CV_8UC1, cv::Scalar(0, 0, 0));
}
Simulator::Simulator() : Simulator(500) {};

/* Getters for the images */
cv::Mat Simulator::getActual() { return imgActual ; }
cv::Mat Simulator::getApparent() { return imgApparent ; }
cv::Mat Simulator::getDistorted() { return imgDistorted ; }

void Simulator::update() {

    auto startTime = std::chrono::system_clock::now();
    
    // Draw the Actual (Source) and Apparent Image
    drawParallel(imgActual, actualX, actualY);
    drawParallel(imgApparent, apparentAbs, 0);
    // The source image has a Gaussian distribution with standard deviation
    // equal to sourceSize.  See drawSource().
    // The apparent image is the source image translated to the apparent position 
    // and to lie on the x axis, i.e. x=r (distance from origin) and y=0.

    this->calculateAlphaBeta() ;

    // Make Distorted Image
    cv::Mat imgD = cv::Mat(imgApparent.size(), CV_8UC1, cv::Scalar(0, 0, 0));
    parallelDistort(imgApparent, imgD);

    // Correct the rotation applied to the source image
    double phi = atan2(actualY, actualX); // Angle relative to x-axis
    cv::Mat rot = cv::getRotationMatrix2D(cv::Point(size, size/2), phi*180/PI, 1);
    cv::warpAffine(imgD, imgD, rot, cv::Size(2*size, size));    // crop distorted image
    imgDistorted =  imgD(cv::Rect(size/2, 0, size, size));

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
    for (int row = begin; row < end; row++) {
        for (int col = 0; col < dst.cols; col++) {
            // if point mass
            int row_, col_;
            std::pair<double, double> pos ;

            // Set coordinate system with origin at x=R
            double x = (col - apparentAbs - dst.cols / 2.0) * CHI;
            double y = (dst.rows / 2.0 - row) * CHI;

            // Calculate distance and angle of the point evaluated relative to center of lens (origin)
            double r = sqrt(x * x + y * y);
            double theta = atan2(y, x);

            pos = this->getDistortedPos(r, theta);

            // Translate to array index
            row_ = (int) round(src.rows / 2.0 - pos.second);
            col_ = (int) round(apparentAbs + src.cols / 2.0 + pos.first);

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
    double x_= (r*cos(theta) + frac * (r / R + cos(theta))) / CHI;
    double y_= (r*sin(theta) - frac * sin(theta)) / CHI;// Point mass lens equation
    return {x_, y_};
}


/* drawParallel() split the image into chunks to draw it in parallel using drawSource() */
void Simulator::drawParallel(cv::Mat& dst, int xPos, int yPos){
    int n_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads_vec;
    for (int i = 0; i < n_threads; i++) {
        int begin = dst.rows / n_threads * i;
        int end = dst.rows / n_threads * (i + 1);
        std::thread t([begin, end, &dst, xPos, yPos, this]() { drawSource(begin, end, dst, xPos, yPos); });
        threads_vec.push_back(std::move(t));
    }
    for (auto& thread : threads_vec) {
        thread.join();
    }
}


/* Draw the source image.  The sourceSize is interpreted as the standard deviation in a Gaussian distribution */
void Simulator::drawSource(int begin, int end, cv::Mat& dst, int xPos, int yPos) {
    for (int row = begin; row < end; row++) {
        for (int col = 0; col < dst.cols; col++) {
            int x = col - xPos - dst.cols/2;
            int y = row + yPos - dst.rows/2;
            auto value = (uchar)round(255 * exp((-x * x - y * y) / (2.0*sourceSize*sourceSize)));
            dst.at<uchar>(row, col) = value;
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

void Simulator::updateSize(double siz) {
   sourceSize = siz ;
   update() ;
}
void Simulator::updateNterms(int n) {
   nterms = n ;
   update() ;
}
void Simulator::updateAll( double X, double Y, double er, double siz, double chi, int n) {
   sourceSize = siz ;
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
    // The two ratioes correspond to two roots of a quadratic equation.
    double ratio1 = 0.5 + 2*sqrt(0.25 + einsteinR*einsteinR/(CHI*CHI*actualAbs*actualAbs));
    double ratio2 = 0.5 - 2*sqrt(0.25 + einsteinR*einsteinR/(CHI*CHI*actualAbs*actualAbs));
    // Each ratio gives rise to one apparent galaxy.
    apparentAbs = actualAbs*ratio1;
    // (X,Y) co-ordinates of first image
    apparentX = actualX*ratio1;
    apparentY = actualY*ratio1;
    // (X,Y) co-ordinates of second image.  This is never used.
    // apparentX2 = actualX*ratio2;
    // apparentY2 = actualY*ratio2;
    // BDN: Is the calculation of apparent positions correct above?

   update() ;
}
/* Default implementation doing nothing.
 * This is correct for any subclass that does not need the alpha/beta tables. */
void Simulator::calculateAlphaBeta() { }
