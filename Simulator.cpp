//
// Created by simon on 07.04.2022.
//

#include "Simulator.h"
#include <symengine/expression.h>
#include <symengine/lambda_double.h>
#include <thread>
#include <symengine/parser.h>
#include <fstream>

#define PI 3.14159265358979323846

double factorial_(unsigned int n);

Simulator::Simulator() :
        size(300),
        CHI_percent(50),
        CHI(CHI_percent/100.0),
        einsteinR(size/20),
        sourceSize(size/20),
        xPosSlider(size/2 + 1),
        yPosSlider(size/2),
        mode(0), // 0 = point mass, 1 = sphere
        n(10)
{

    GAMMA = einsteinR/2.0;
}


void Simulator::update() {

    auto startTime = std::chrono::system_clock::now();

    GAMMA = einsteinR/2.0;
    calculate();
    cv::Mat imgActual(size, size, CV_8UC1, cv::Scalar(0, 0, 0));
//    cv::circle(imgActual, cv::Point(size/2 + (int)actualX, size/2 - (int)actualY), sourceSize, cv::Scalar::all(255), 2*sourceSize);
    drawParallel(imgActual, actualX, actualY);
    cv::Mat imgApparent;

    // if point
    if (mode == 0){
        imgApparent = cv::Mat(size, 2*size, CV_8UC1, cv::Scalar(0, 0, 0));
        imgDistorted = cv::Mat(imgApparent.size(), CV_8UC1, cv::Scalar(0, 0, 0));
//        cv::circle(imgApparent, cv::Point(size + (int)apparentAbs, size/2), sourceSize, cv::Scalar::all(255), 2*sourceSize);
        drawParallel(imgApparent, apparentAbs, 0);
        parallelDistort(imgApparent, imgDistorted);
        // rotate image
        cv::Mat rot = cv::getRotationMatrix2D(cv::Point(size, size/2), phi*180/PI, 1);
        cv::warpAffine(imgDistorted, imgDistorted, rot, cv::Size(2*size, size));    // crop distorted image
        imgDistorted =  imgDistorted(cv::Rect(size/2, 0, size, size));
    }

    // if Spherical
    else if (mode == 1){

        // calculate all amplitudes for given X, Y, GAMMA, CHI
        for (int m = 1; m <= n; m++){
            for (int s = (m+1)%2; s <= (m+1); s+=2){
                alphas_val[m][s] = alphas_l[m][s].call({X, Y, GAMMA, CHI});
                betas_val[m][s] = betas_l[m][s].call({X, Y, GAMMA, CHI});
            }
        }

        imgApparent = cv::Mat(2*size, 2*size, CV_8UC1, cv::Scalar(0, 0, 0));
        imgDistorted = cv::Mat(size, size, CV_8UC1, cv::Scalar(0, 0, 0));
        cv::circle(imgApparent, cv::Point(size + (int)apparentX, size - (int)apparentY), sourceSize, cv::Scalar::all(255), 2*sourceSize);
//        cv::imshow("apparent", imgApparent);
        parallelDistort(imgApparent, imgDistorted);
    }

//    cv::cvtColor(imgDistorted, imgDistorted, cv::COLOR_GRAY2BGR);

//    const int displaySize = 500;
//    refLines(imgActual);
//    refLines(imgDistorted);
//    cv::Mat matDst = formatImg(imgDistorted, imgActual, displaySize);
    cv::Mat matDst(cv::Size(2*size, size), imgActual.type(), cv::Scalar::all(255));
    cv::Mat matRoi = matDst(cv::Rect(0, 0, size, size));
    imgActual.copyTo(matRoi);
    matRoi = matDst(cv::Rect(size, 0, size, size));
    imgDistorted.copyTo(matRoi);

    cv::imshow("GL Simulator", matDst);

    auto endTime = std::chrono::system_clock::now();
//    std::cout << "Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count() << " milliseconds" << std::endl;

}


void Simulator::parallelDistort(const cv::Mat& src, cv::Mat& dst) {
    int n_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads_vec;
    for (int i = 0; i < n_threads; i++) {
        int begin = dst.rows/n_threads*i;
        int end = dst.rows/n_threads*(i+1);
            std::thread t([begin, end, src, &dst, this]() { distort(begin, end, src, dst); });
            threads_vec.push_back(std::move(t));
//            distort(row, col, src, dst);
        }
//        if(mode){
//            std::cout << 100.0 * row / dst.rows << "%" << std::endl;
//        }

    for (auto& thread : threads_vec) {
        thread.join();
    }
}


void Simulator::distort(int begin, int end, const cv::Mat& src, cv::Mat& dst) {
    for (int row = begin; row < end; row++) {
        for (int col = 0; col < dst.cols; col++) {
            // if point mass
            int row_, col_;

            if (!mode) {
                // Set coordinate system with origin at x=R
                double x = (col - apparentAbs - dst.cols / 2.0) * CHI;
                double y = (dst.rows / 2.0 - row) * CHI;
                // Calculate distance and angle of the point evaluated relative to center of lens (origin)
                double r = sqrt(x * x + y * y);
                double theta = atan2(y, x);
                auto pos = pointMass(r, theta);
                // Translate to array index
                row_ = (int) round(src.rows / 2.0 - pos.second);
                col_ = (int) round(apparentAbs + src.cols / 2.0 + pos.first);
            }
                // if sphere
            else {

                double x = col - dst.cols / 2.0 - X;
                double y = dst.rows / 2.0 - row - Y;
                double r = sqrt(x * x + y * y);

                double theta = atan2(y, x);
                auto pos = spherical(r, theta);
                // Translate to array index
                col_ = (int) round(apparentX + src.cols / 2.0 + pos.first);
                row_ = (int) round(src.rows / 2.0 - pos.second - apparentY);
            }

            // If (x', y') within source, copy value to imgDistorted
            if (row_ < src.rows && col_ < src.cols && row_ >= 0 && col_ >= 0) {
                auto val = src.at<uchar>(row_, col_);
                dst.at<uchar>(row, col) = val;
            }
        }
    }
}

//for (int m = 1; m <= n; m++){
//for (int s = (m+1)%2; s <= (m+1); s+=2){

std::pair<double, double> Simulator::spherical(double r, double theta) const {
    double ksi1 = 0;
    double ksi2 = 0;

    for (int m=1; m<=n; m++){
        double frac = pow(r, m) / factorial_(m);
        double subTerm1 = 0;
        double subTerm2 = 0;
        for (int s = (m+1)%2; s <= (m+1); s+=2){
            double alpha = alphas_val[m][s];
            double beta = betas_val[m][s];
            int c_p = 1 + s/(m + 1);
            int c_m = 1 - s/(m + 1);
            subTerm1 += 1.0/4*((alpha*cos((s-1)*theta) + beta*sin((s-1)*theta))*c_p + (alpha*cos((s+1)*theta) + beta*sin((s+1)*theta))*c_m);
            subTerm2 += 1.0/4*((-alpha*sin((s-1)*theta) + beta*cos((s-1)*theta))*c_p + (alpha*sin((s+1)*theta) - beta*cos((s+1)*theta))*c_m);
        }
        double term1 = frac*subTerm1;
        double term2 = frac*subTerm2;
        ksi1 += term1;
        ksi2 += term2;
        // Break summation if term is less than 1/100 of ksi or if ksi is well outside frame
//        if ( ((std::abs(term1) < std::abs(ksi1)/1000) && (std::abs(term2) < std::abs(ksi2)/1000)) || (ksi1 < -1000*size || ksi1 > 1000*size || ksi2 < -1000*size || ksi2 > 1000*size) ){
//            break;
//        }
    }
    return {ksi1, ksi2};
}


std::pair<double, double> Simulator::pointMass(double r, double theta) const {
    double frac = (einsteinR * einsteinR * r) / (r * r + R * R + 2 * r * R * cos(theta));
    double x_= (r*cos(theta) + frac * (r / R + cos(theta))) / CHI;
    double y_= (r*sin(theta) - frac * sin(theta)) / CHI;// Point mass lens equation
    return {x_, y_};
}


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


void Simulator::writeToPngFiles(int n_params) {
    std::ostringstream filename_path;
    std::ostringstream filename;

    if (n_params == 5){
        filename << CHI_percent << ",";
    }
    filename << einsteinR << "," << sourceSize << "," << xPosSlider << "," << yPosSlider << ".png";
    filename_path << name + "/images/" + filename.str();
    cv::imwrite(filename_path.str(), imgDistorted);
//    cv::imshow(filename_path.str(), image);
//    cv::waitKey(0);
}


double factorial_(unsigned int n){
    double a = 1.0;
    for (int i = 2; i <= n; i++){
        a *= i;
    }
    return a;
}


void Simulator::calculate() {

    CHI = CHI_percent/100.0;

    // actual position in source plane
    actualX = xPosSlider - size / 2.0;
    actualY = yPosSlider - size / 2.0;

    // Absolute values in source plane
    actualAbs = sqrt(actualX * actualX + actualY * actualY);
    double ratio1 = 0.5 + sqrt(0.25 + einsteinR*einsteinR/(CHI*CHI*actualAbs*actualAbs));
    double ratio2 = 0.5 - sqrt(0.25 + einsteinR*einsteinR/(CHI*CHI*actualAbs*actualAbs));
    apparentAbs = actualAbs*ratio1;
    apparentAbs2 = actualAbs*ratio2;
    apparentX = actualX*ratio1;
    apparentX2 = actualX*ratio2;
    apparentY = actualY*ratio1;
    apparentY2 = actualY*ratio2;

    // Projection of apparent position in lens plane
    R = apparentAbs * CHI;
    X = apparentX * CHI;
    Y = apparentY * CHI;

    // Angle relative to x-axis
    phi = atan2(actualY, actualX);

}


//cv::Mat Simulator::formatImg(cv::Mat &imgDistorted, cv::Mat &imgActual, int displaySize) const {
//    if (!mode){
//        cv::circle(imgDistorted, cv::Point(size / 2, size / 2), (int)round(einsteinR / CHI), cv::Scalar::all(60));
//        cv::drawMarker(imgDistorted, cv::Point(size / 2 + (int) apparentX, size / 2 - (int) apparentY), cv::Scalar(0, 0, 255), cv::MARKER_TILTED_CROSS, size / 30);
//        cv::drawMarker(imgDistorted, cv::Point(size / 2 + (int) apparentX2, size / 2 - (int) apparentY2), cv::Scalar(0, 0, 255), cv::MARKER_TILTED_CROSS, size / 30);
//        cv::drawMarker(imgDistorted, cv::Point(size / 2 + (int) actualX, size / 2 - (int) actualY), cv::Scalar(255, 0, 0), cv::MARKER_TILTED_CROSS, size / 30);
//    }
//    cv::resize(imgActual, imgActual, cv::Size(displaySize, displaySize));
//    cv::resize(imgDistorted, imgDistorted, cv::Size(displaySize, displaySize));
//    cv::Mat matDst(cv::Size(2*displaySize, displaySize), imgActual.type(), cv::Scalar::all(255));
//    cv::Mat matRoi = matDst(cv::Rect(0, 0, displaySize, displaySize));
//    imgActual.copyTo(matRoi);
//    matRoi = matDst(cv::Rect(displaySize, 0, displaySize, displaySize));
//    imgDistorted.copyTo(matRoi);
//    return matDst;
//}


//// Add some lines to the image for reference
//void Simulator::refLines(cv::Mat& target) {
//    int size_ = target.rows;
//    for (int i = 0; i < size_; i++) {
//        target.at<cv::Vec3b>(i, size_ / 2) = {60, 60, 60};
//        target.at<cv::Vec3b>(size_ / 2 - 1, i) = {60, 60, 60};
//        target.at<cv::Vec3b>(i, size_ - 1) = {255, 255, 255};
//        target.at<cv::Vec3b>(i, 0) = {255, 255, 255};
//        target.at<cv::Vec3b>(size_ - 1, i) = {255, 255, 255};
//        target.at<cv::Vec3b>(0, i) = {255, 255, 255};
//    }
//}


void Simulator::initGui(){
    initAlphasBetas();
    // Make the user interface and specify the function to be called when moving the sliders: update()
    cv::namedWindow("GL Simulator", cv::WINDOW_AUTOSIZE);
    cv::createTrackbar("Lens dist %    :", "GL Simulator", &CHI_percent, 100, update_dummy, this);
    cv::createTrackbar("Einstein radius / Gamma:", "GL Simulator", &einsteinR, size, update_dummy, this);
    cv::createTrackbar("Source sourceSize   :", "GL Simulator", &sourceSize, size / 10, update_dummy, this);
    cv::createTrackbar("X position     :", "GL Simulator", &xPosSlider, size, update_dummy, this);
    cv::createTrackbar("Y position     :", "GL Simulator", &yPosSlider, size, update_dummy, this);
    cv::createTrackbar("\t\t\t\t\t\t\t\t\t\tMode, point/sphere:\t\t\t\t\t\t\t\t\t\t", "GL Simulator", &mode, 1, update_dummy, this);
    cv::createTrackbar("sum from m=1 to...:", "GL Simulator", &n, 49, update_dummy, this);
}


void Simulator::update_dummy(int, void* data){
    auto* that = (Simulator*)(data);
    that->update();
}


void Simulator::initAlphasBetas() {

    auto x = SymEngine::symbol("x");
    auto y = SymEngine::symbol("y");
    auto g = SymEngine::symbol("g");
    auto c = SymEngine::symbol("c");

    std::string filename("../../50.txt");
    std::ifstream input;
    input.open(filename);

    if (!input.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }

    while (input) {
        std::string m, s;
        std::string alpha;
        std::string beta;
        std::getline(input, m, ':');
        std::getline(input, s, ':');
        std::getline(input, alpha, ':');
        std::getline(input, beta);
        if (input) {
            auto alpha_sym = SymEngine::parse(alpha);
            auto beta_sym = SymEngine::parse(beta);
            SymEngine::LambdaRealDoubleVisitor alpha_num, beta_num;
            alphas_l[std::stoi(m)][std::stoi(s)].init({x, y, g, c}, *alpha_sym);
            betas_l[std::stoi(m)][std::stoi(s)].init({x, y, g, c}, *beta_sym);
        }
    }
}

