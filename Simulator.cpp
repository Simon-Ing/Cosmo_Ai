//
// Created by simon on 07.04.2022.
//

#include "Simulator.h"
#include <symengine/expression.h>
#include <symengine/lambda_double.h>
#include <thread>

#define PI 3.14159265358979323846

double factorial_(unsigned int n);
class Timer;

Simulator::Simulator() :
    size(60),
    CHI_percent(50),
    CHI(CHI_percent/100.0),
    einsteinR(size/20),
    GAMMA(einsteinR),
    sourceSize(size/20),
    xPosSlider(size/2 + 1),
    yPosSlider(size/2 +1),
    mode(0) // 0 = point mass, 1 = sphere
{
}

void Simulator::initGui(){
    initAlphasBetas(alphas_l, betas_l);
    // Make the user interface and specify the function to be called when moving the sliders: update()
    cv::namedWindow("GL Simulator", cv::WINDOW_AUTOSIZE);
    cv::createTrackbar("Lens dist %    :", "GL Simulator", &CHI_percent, 100, update_dummy, this);
    cv::createTrackbar("Einstein radius:", "GL Simulator", &einsteinR, size, update_dummy, this);
    cv::createTrackbar("Source sourceSize   :", "GL Simulator", &sourceSize, size / 10, update_dummy, this);
    cv::createTrackbar("X position     :", "GL Simulator", &xPosSlider, size, update_dummy, this);
    cv::createTrackbar("Y position     :", "GL Simulator", &yPosSlider, size, update_dummy, this);
    cv::createTrackbar("Mode, point/sphere (in sphere mode: set sliders, then hit space to run):", "GL Simulator", &mode, 1, update_dummy, this);
}


void Simulator::update_dummy(int, void* data){
    auto* that = (Simulator*)(data);
    if (!that->mode){ // if point mass mode
        that->update();
    }
}


void Simulator::initAlphasBetas(std::array<std::array<LambdaRealDoubleVisitor, n>, n>& alphas_lambda, std::array<std::array<LambdaRealDoubleVisitor, n>, n>& betas_lambda) {
    xSym = symbol("x");
    ySym = symbol("y");
    gammaSym = symbol("gamma");
    chiSym = symbol("chi_l");
    alphas = std::vector<std::vector<SymEngine::RCP<const Basic>>>(n, std::vector<SymEngine::RCP<const Basic>>(n));
    betas = std::vector<std::vector<SymEngine::RCP<const Basic>>> (n, std::vector<SymEngine::RCP<const Basic>>(n));
    auto psi = sqrt(add(mul(xSym, xSym), mul(ySym, ySym)));
    alphas[0][1] = diff(psi, xSym, false);
    betas[0][1] = neg(diff(psi, ySym, false));

    // First diagonal
    for (int m = 1; m<n-1; m++){
        int s = m + 1;
        double c_num = (m + 1.0)/(m + 1.0 + s)*(1 + (s == 1));
        auto c = mul(real_double(c_num), chiSym);
        alphas[m][s] = mul(c, sub(diff(alphas[m-1][s-1], xSym, false), diff(betas[m-1][s-1], ySym, false)));
        betas[m][s] = mul(c, add(diff(betas[m-1][s-1], xSym, false), diff(alphas[m-1][s-1], ySym, false)));
        alphas_lambda[m][s].init({xSym, ySym, gammaSym, chiSym}, *alphas[m][s]);
        betas_lambda[m][s].init({xSym, ySym, gammaSym, chiSym}, *betas[m][s]);
    }

    // The rest
    for (int start = 1; start < n; start+=2){
        for(int s=0; s<(n-start); s++){
            int m = s + start;
            double c_num = (m + 1.0)/(m + 1.0 - s)*(1.0 + (s != 0)/2.0);
            auto c = mul(real_double(c_num), chiSym);
            alphas[m][s] = mul(c, add(diff(alphas[m-1][s+1], xSym, false), diff(betas[m-1][s+1], ySym, false)));
            betas[m][s] = mul(c, sub(diff(betas[m-1][s+1], xSym, false), diff(alphas[m-1][s+1], ySym, false)));
            alphas_lambda[m][s].init({xSym, ySym, gammaSym, chiSym}, *alphas[m][s]);
            betas_lambda[m][s].init({xSym, ySym, gammaSym, chiSym}, *betas[m][s]);
        }
    }
}

void Simulator::update() {

    auto startTime = std::chrono::system_clock::now();

    GAMMA = einsteinR;
    calculate();
    cv::Mat imgApparent(size, 2*size, CV_8UC1, cv::Scalar(0, 0, 0));
    cv::Mat imgDistorted(imgApparent.size(), CV_8UC1, cv::Scalar(0, 0, 0));
    // make an image with light source at ACTUAL position
    cv::Mat imgActual(size, size, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::circle(imgActual, cv::Point(size/2 + (int)actualX, size/2 - (int)actualY), sourceSize, cv::Scalar::all(255), 2*sourceSize);

    // if point
    if (mode == 0){
        cv::circle(imgApparent, cv::Point(size + (int)apparentAbs, size/2), sourceSize, cv::Scalar::all(255), 2*sourceSize);
        parallelDistort(imgApparent, imgDistorted);
        // rotate image
        cv::Mat rot = cv::getRotationMatrix2D(cv::Point(size, size/2), phi*180/PI, 1);
        cv::warpAffine(imgDistorted, imgDistorted, rot, cv::Size(2*size, size));
    }
    // if Spherical
    else if (mode == 1){
        cv::circle(imgApparent, cv::Point(size + (int)actualX, size/2 - (int)actualY), sourceSize, cv::Scalar::all(255), 2*sourceSize);
        parallelDistort(imgApparent, imgDistorted);
//        distort(0, size, imgActual, imgDistorted);
    }

    // crop distorted image
    imgDistorted =  imgDistorted(cv::Rect(size/2, 0, size, size));
    cv::cvtColor(imgDistorted, imgDistorted, cv::COLOR_GRAY2BGR);


    int displaySize = 600;

    refLines(imgActual);
    refLines(imgDistorted);

    cv::Mat imgOutput;

    cv::Mat matDst = formatImg(imgDistorted, imgActual, displaySize);

    cv::imshow("GL Simulator", matDst);

    auto endTime = std::chrono::system_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count() << std::endl;
    // 6.5s 60 10 laptop
}

// Add some lines to the image for reference
void Simulator::refLines(cv::Mat& target) {
    int size_ = target.rows;
    for (int i = 0; i < size_; i++) {
        target.at<cv::Vec3b>(i, size_ / 2) = {60, 60, 60};
        target.at<cv::Vec3b>(size_ / 2 - 1, i) = {60, 60, 60};
        target.at<cv::Vec3b>(i, size_ - 1) = {255, 255, 255};
        target.at<cv::Vec3b>(i, 0) = {255, 255, 255};
        target.at<cv::Vec3b>(size_ - 1, i) = {255, 255, 255};
        target.at<cv::Vec3b>(0, i) = {255, 255, 255};
    }
}

cv::Mat Simulator::formatImg(cv::Mat &imgDistorted, cv::Mat &imgActual, int displaySize) const {
    cv::circle(imgDistorted, cv::Point(size / 2, size / 2), (int)round(einsteinR / CHI), cv::Scalar::all(60));
    cv::drawMarker(imgDistorted, cv::Point(size / 2 + (int) apparentX, size / 2 - (int) apparentY), cv::Scalar(0, 0, 255), cv::MARKER_TILTED_CROSS, size / 30);
    cv::drawMarker(imgDistorted, cv::Point(size / 2 + (int) apparentX2, size / 2 - (int) apparentY2), cv::Scalar(0, 0, 255), cv::MARKER_TILTED_CROSS, size / 30);
    cv::drawMarker(imgDistorted, cv::Point(size / 2 + (int) actualX, size / 2 - (int) actualY), cv::Scalar(255, 0, 0), cv::MARKER_TILTED_CROSS, size / 30);
    cv::resize(imgActual, imgActual, cv::Size(displaySize, displaySize));
    cv::resize(imgDistorted, imgDistorted, cv::Size(displaySize, displaySize));
    cv::Mat matDst(cv::Size(2*displaySize, displaySize), imgActual.type(), cv::Scalar::all(255));
    cv::Mat matRoi = matDst(cv::Rect(0, 0, displaySize, displaySize));
    imgActual.copyTo(matRoi);
    matRoi = matDst(cv::Rect(displaySize, 0, displaySize, displaySize));
    imgDistorted.copyTo(matRoi);
    return matDst;
}

void Simulator::calculate() {

    CHI = CHI_percent/100.0;
    // actual position in source plane
    actualX = xPosSlider - size / 2.0;
    actualY = yPosSlider - size / 2.0;

    // Absolute values in source plane
    actualAbs = sqrt(actualX * actualX + actualY * actualY);
    apparentAbs = (actualAbs + sqrt(actualAbs * actualAbs + 4 / (CHI * CHI) * einsteinR * einsteinR)) / 2.0;
    apparentAbs2 = (actualAbs - sqrt(actualAbs * actualAbs + 4 / (CHI * CHI) * einsteinR * einsteinR)) / 2.0;

    // Apparent position in source plane
    apparentX = actualX * apparentAbs / actualAbs;
    apparentY = actualY * apparentAbs / actualAbs;
    apparentX2 = actualX * apparentAbs2 / actualAbs;
    apparentY2 = actualY * apparentAbs2 / actualAbs;


    // Projection of apparent position in lens plane
    R = apparentAbs * CHI;
    X = apparentX * CHI;
    Y = apparentY * CHI;

    // Angle relative to x-axis
    phi = atan2(actualY, actualX);
}

// Split the image into (number of threads available) pieces and distort the pieces in parallel
void Simulator::parallelDistort(const cv::Mat& src, cv::Mat& dst) {
    unsigned int num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads_vec;
    for (int k = 0; k < num_threads; k++) {
        int thread_begin = (int)(src.rows / num_threads) * k;
        int thread_end = (int)(src.rows / num_threads) * (k + 1);
        std::thread t([thread_begin, thread_end, src, &dst, this](){distort(thread_begin, thread_end, src, dst);});
        threads_vec.push_back(std::move(t));
    }
    for (auto& thread : threads_vec) {
        thread.join();
    }
}

void Simulator::distort(int begin, int end, const cv::Mat& src, cv::Mat& dst) {
    // Evaluate each point in imgDistorted plane ~ lens plane
    for (int row = begin; row < end; row++) {
        for (int col = 0; col < dst.cols; col++) {

            int row_, col_;

            // if point mass
            if (!mode){

                // Set coordinate system with origin at x=R
                double x = (col - apparentAbs - dst.cols / 2.0) * CHI;
                double y = (dst.rows/2.0 - row) * CHI;

                // Calculate distance and angle of the point evaluated relative to center of lens (origin)
                double r = sqrt(x*x + y*y);
                double theta = atan2(y, x);
                auto pos = pointMass(r, theta);

                // Translate to array index
                row_ = (int)round(src.rows / 2.0 - pos.second);
                col_ = (int)round(apparentAbs + src.cols / 2.0 + pos.first);
            }

            // if sphere
            else{
                // Set coordinate system with origin at x=R
                double x = (col - actualX - dst.cols / 2.0) * CHI;
                double y = (dst.rows/2.0 - row - actualY) * CHI;
                // Calculate distance and angle of the point evaluated relative to center of lens (origin)
                double r = sqrt(x*x + y*y);
                double theta = atan2(y, x);
                auto pos = spherical(r, theta, alphas_l, betas_l);

                // Translate to array index
                row_ = (int)round(src.rows / 2.0 - pos.second - actualY);
                col_ = (int)round(apparentAbs + src.cols / 2.0 + pos.first);
            }


            // If (x', y') within source, copy value to imgDistorted
            if (row_ < src.rows && col_ < src.cols && row_ >= 0 && col_ >= 0) {
                dst.at<uchar>(row, col) = src.at<uchar>(row_, col_);
            }
        }
    }
}

std::pair<double, double> Simulator::pointMass(double r, double theta) const {
    double frac = (einsteinR * einsteinR * r) / (r * r + R * R + 2 * r * R * cos(theta));
    double x_= (r*cos(theta) + frac * (r / R + cos(theta))) / CHI;
    double y_= (r*sin(theta) - frac * sin(theta)) / CHI;// Point mass lens equation
    return {x_, y_};
}

std::pair<double, double> Simulator::spherical(double r, double theta, std::array<std::array<LambdaRealDoubleVisitor, n>, n>& alphas_lambda, std::array<std::array<LambdaRealDoubleVisitor, n>, n>& betas_lambda) const {
    double ksi1 = 0;
    double ksi2 = 0;

    int count = 0;
    double max1 = 0;
    double max2 = 0;

    for (int m=1; m<n; m++){
        double frac = pow(r, m) / factorial_(m);
        double subTerm1 = 0;
        double subTerm2 = 0;
        for (int s=(m+1)%2; s<=m+1 && s<n; s+=2){
            count ++;
            double alpha = GAMMA/(CHI) * alphas_lambda[m][s].call({X, Y, GAMMA, CHI});
            double beta = GAMMA/(CHI) * betas_lambda[m][s].call({X, Y, GAMMA, CHI});
            int c_p = 1 + s/(m + 1);
            int c_m = 1 - s/(m + 1);
            subTerm1 += theta*((alpha*cos(s-1) + beta*sin(s-1))*c_p + (alpha*cos(s+1) + beta*sin(s+1)*c_m));
            subTerm2 += theta*((-alpha*sin(s-1) + beta*cos(s-1))*c_p + (alpha*sin(s+1) - beta*cos(s+1)*c_m));
        }
        double term1 = frac*subTerm1;
        double term2 = frac*subTerm2;
        ksi1 += term1;
        ksi2 += term2;
        max1 = std::max(std::abs(term1), max1);
        max2 = std::max(std::abs(term2), max2);
//        std::cout << "Count: " << count << " m: " << m << " term 1: " << term1 << " term 2 " << term2 << " max1: " << max1 << " max2: " << max2 << std::endl;
        if ( std::abs(term1) < max1/100 && std::abs(term2) < max2/100){
            break;
        }
    }
    return {ksi1, ksi2};
}


double factorial_(unsigned int n){
    double a = 1.0;
    for (int i = 2; i <= n; i++){
        a *= i;
    }
    return a;
}