//
// Created by simon on 07.04.2022.
//

#include "Simulator.h"
#include <symengine/expression.h>
#include <symengine/lambda_double.h>
#include <thread>
#include <fstream>

#define PI 3.14159265358979323846


double factorial_(unsigned int n);

Simulator::Simulator() :
    size(80),
    CHI_percent(50),
    CHI(CHI_percent/100.0),
    einsteinR(size/20),
    GAMMA(einsteinR),
    sourceSize(size/20),
    xPosSlider(size/2 + 1),
    yPosSlider(size/2),
    X(0),
    Y(0),
    mode(1) // 0 = point mass, 1 = sphere
{
}


void Simulator::update() {

    auto startTime = std::chrono::system_clock::now();

    GAMMA = einsteinR;
    calculate();
    cv::Mat imgApparent(size, 2*size, CV_8UC1, cv::Scalar(100, 100, 100));
    cv::Mat imgDistorted(imgApparent.size(), CV_8UC1, cv::Scalar(0, 0, 0));
    // make an image with light source at ACTUAL position
    cv::Mat imgActual(size, size, CV_8UC3, cv::Scalar(100, 100, 100));
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
        cv::circle(imgApparent, cv::Point(size + (int)apparentX, size/2 - (int)apparentY), sourceSize, cv::Scalar::all(255), 2*sourceSize);
        cv::imshow("apparent", imgApparent);
        parallelDistort(imgApparent, imgDistorted);
//        distort(0, size, imgApparent, imgDistorted);
    }

    // crop distorted image
    imgDistorted =  imgDistorted(cv::Rect(size/2, 0, size, size));
    cv::cvtColor(imgDistorted, imgDistorted, cv::COLOR_GRAY2BGR);

    const int displaySize = 500;
    refLines(imgActual);
    refLines(imgDistorted);
    cv::Mat imgOutput;
    cv::Mat matDst = formatImg(imgDistorted, imgActual, displaySize);
    cv::imshow("GL Simulator", matDst);

    auto endTime = std::chrono::system_clock::now();
    std::cout << "Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count() << std::endl;

}


void Simulator::parallelDistort(const cv::Mat& src, cv::Mat& dst) {
    std::vector<std::thread> threads_vec;
    std::mutex m;
    for (int row = 0; row < dst.rows; row++) {
        for (int col = 0; col < dst.cols; col++) {
//            std::thread t([row, col, src, &dst, &m, this]() { distort(row, col, src, dst, m); });
//            threads_vec.push_back(std::move(t));
            std::cout << (double)row / dst.rows << "% init" << std::endl;
            distort(row, col, src, dst, m);
        }
    }
    double i = 0;
    for (auto& thread : threads_vec) {
        i++;
        std::cout << i / (dst.cols*dst.rows) * 100.0 << "%" << std::endl;
        thread.join();
    }
}


void Simulator::distort(int row, int col, const cv::Mat& src, cv::Mat& dst, std::mutex& m) {
    // Evaluate each point in imgDistorted plane ~ lens plane
    int row_, col_;
    // Set coordinate system with origin at x=R
    double x = (col - apparentAbs - dst.cols / 2.0) * CHI;
    double y = (dst.rows / 2.0 - row) * CHI;
    // Calculate distance and angle of the point evaluated relative to center of lens (origin)
    double r = sqrt(x * x + y * y);
    double theta = atan2(y, x);
    // if point mass
    if (!mode) {
        auto pos = pointMass(r, theta);
        // Translate to array index
        row_ = (int) round(src.rows / 2.0 - pos.second);
        col_ = (int) round(apparentAbs + src.cols / 2.0 + pos.first);
    }
        // if sphere
    else {
        auto pos = spherical(r, theta, m);
        // Translate to array index
        row_ = (int) round(src.rows / 2.0 - pos.second - apparentY);
        col_ = (int) round(apparentX + src.cols / 2.0 + pos.first);
    }

    // If (x', y') within source, copy value to imgDistorted
    if (row_ < src.rows && col_ < src.cols && row_ >= 0 && col_ >= 0) {
//        std::cout << "row: " << row << " col: " << col << " row_: " << row_ << " col_: " << col_ << std::endl;
        auto val = src.at<uchar>(row_, col_);
        dst.at<uchar>(row, col) = val;
    }
}


std::pair<double, double> Simulator::spherical(double r, double theta, std::mutex& mut) const {

    double ksi1 = 0;
    double ksi2 = 0;

    for (int m=1; m<n; m++){
        double frac = pow(r, m) / factorial_(m);
        double subTerm1 = 0;
        double subTerm2 = 0;
        for (int s=(m+1)%2; s<=m+1 && s<n; s+=2) {
            GiNaC::ex alpha_eval, beta_eval;

//                std::lock_guard<std::mutex> lock(mut);
            alpha_eval = GiNaC::evalf(alphas[m][s].subs(GiNaC::lst{x == X, y == Y, c == CHI, g == GAMMA}));
            beta_eval = GiNaC::evalf(betas[m][s].subs(GiNaC::lst{x == X, y == Y, c == CHI, g == GAMMA}));


            auto alpha_num = GiNaC::ex_to<GiNaC::numeric>(alpha_eval).to_double();
            auto beta_num = GiNaC::ex_to<GiNaC::numeric>(beta_eval).to_double();

            //            std::cout << "\nm: " << m << " s: " << s << "\nalpha: " << alpha << " beta: " << beta <<"\nalpha: " << alpha << " beta: " << beta << std::endl;
            auto c_p = 1 + s / (m + 1);
            auto c_m = 1 - s / (m + 1);

            subTerm1 += theta * ((alpha_num * cos(s - 1) + beta_num * sin(s - 1)) * c_p +
                                 (alpha_num * cos(s + 1) + beta_num * sin(s + 1) * c_m));
            subTerm2 += theta * ((-alpha_num * sin(s - 1) + beta_num * cos(s - 1)) * c_p +
                                 (alpha_num * sin(s + 1) - beta_num * cos(s + 1) * c_m));

        }
//        std::cout << "1" << std::endl;
        double term1 = frac*subTerm1;
        double term2 = frac*subTerm2;
//        std::cout << "2" << std::endl;
        ksi1 += term1;
        ksi2 += term2;
//        std::cout << "3" << std::endl;
        if ( std::abs(term1) < std::abs(ksi1)/100 && std::abs(term2) < std::abs(ksi2)/100){
            break;
        }
//        std::cout << "4" << std::endl;
    }
//    std::cout << "5" << std::endl;
//    std::cout << "ksi1: " << ksi1 << " ksi2: " << ksi2 << std::endl;
    return {ksi1, ksi2};
}


std::pair<double, double> Simulator::pointMass(double r, double theta) const {
    double frac = (einsteinR * einsteinR * r) / (r * r + R * R + 2 * r * R * cos(theta));
    double x_= (r*cos(theta) + frac * (r / R + cos(theta))) / CHI;
    double y_= (r*sin(theta) - frac * sin(theta)) / CHI;// Point mass lens equation
    return {x_, y_};
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

    std::cout << "actualX: " << actualX << " actualY: " << actualY << " actualAbs: " << actualAbs <<
              " apparentX: " <<apparentX << " apparentY: " << apparentY << " apparentAbs: " << apparentAbs <<
              " R:" << R << " X: " << X << " Y: " << Y << std::endl;
}


cv::Mat Simulator::formatImg(cv::Mat &imgDistorted, cv::Mat &imgActual, int displaySize) const {
    cv::circle(imgDistorted, cv::Point(size / 2, size / 2), (int)round(einsteinR / CHI), cv::Scalar::all(60));
    if (!mode){
        cv::drawMarker(imgDistorted, cv::Point(size / 2 + (int) apparentX, size / 2 - (int) apparentY), cv::Scalar(0, 0, 255), cv::MARKER_TILTED_CROSS, size / 30);
        cv::drawMarker(imgDistorted, cv::Point(size / 2 + (int) apparentX2, size / 2 - (int) apparentY2), cv::Scalar(0, 0, 255), cv::MARKER_TILTED_CROSS, size / 30);
        cv::drawMarker(imgDistorted, cv::Point(size / 2 + (int) actualX, size / 2 - (int) actualY), cv::Scalar(255, 0, 0), cv::MARKER_TILTED_CROSS, size / 30);
    }
    cv::resize(imgActual, imgActual, cv::Size(displaySize, displaySize));
    cv::resize(imgDistorted, imgDistorted, cv::Size(displaySize, displaySize));
    cv::Mat matDst(cv::Size(2*displaySize, displaySize), imgActual.type(), cv::Scalar::all(255));
    cv::Mat matRoi = matDst(cv::Rect(0, 0, displaySize, displaySize));
    imgActual.copyTo(matRoi);
    matRoi = matDst(cv::Rect(displaySize, 0, displaySize, displaySize));
    imgDistorted.copyTo(matRoi);
    return matDst;
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


void Simulator::initGui(){
    initAlphasBetas();
    // Make the user interface and specify the function to be called when moving the sliders: update()
    cv::namedWindow("GL Simulator", cv::WINDOW_AUTOSIZE);
    cv::createTrackbar("Lens dist %    :", "GL Simulator", &CHI_percent, 100, update_dummy, this);
    cv::createTrackbar("Einstein radius / Gamma:", "GL Simulator", &einsteinR, size, update_dummy, this);
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


void Simulator::initAlphasBetas() {

    x = GiNaC::symbol("x");
    y = GiNaC::symbol("y");
    c = GiNaC::symbol("c");
    g = GiNaC::symbol("g");

    syms = {x, y, c, g};

    int n_ = 20;

//    alphas = std::array<std::array<GiNaC::ex>>(n_, std::array<GiNaC::ex>(n_));
//    betas = std::array<std::array<GiNaC::ex>>(n_, std::array<GiNaC::ex>(n_));

    std::string filename("../../functions.txt");
    std::ifstream input;
    input.open(filename);

    if(!input.is_open()){
        std::cout << "Could not open functions.txt file" << std::endl;
    }

    while(input){
        std::string m, s;
        std::string alpha;
        std::string beta;
        std::getline(input, m, ':');
        std::getline(input, s, ':');
        std::getline(input, alpha, ':');
        std::getline(input, beta);
        if(input){
//            std::cout << std::stoi(m) << " " << std::stoi(s) << " " << alpha << " " << beta << std::endl;
            GiNaC::ex a(alpha, syms);
            GiNaC::ex b(beta, syms);
            alphas[std::stoi(m)][std::stoi(s)] = a;
            betas[std::stoi(m)][std::stoi(s)] = b;
//            std::cout << std::endl;
        }
    }
    input.close();

    /*Make matrices containing the symbolic functions and numeric functions for all alpha_m_s and beta_m_s at index [m][s]*/

    // Symbolic variables
//    xSym = symbol("x");
//    ySym = symbol("y");
//    gammaSym = symbol("gamma");
//    chiSym = symbol("chi_l");



    // Symbolic functions
//    alphas = std::vector<std::vector<SymEngine::RCP<const Basic>>>(n, std::vector<SymEngine::RCP<const Basic>>(n));
//    betas = std::vector<std::vector<SymEngine::RCP<const Basic>>> (n, std::vector<SymEngine::RCP<const Basic>>(n));
//    auto psi = mul(div( mul(integer(2), gammaSym), mul(chiSym, chiSym)) , sqrt(add(mul(xSym, xSym), mul(ySym, ySym))));




//    // Calculate and insert the first alpha and beta
//    alphas[0][1] = mul(chiSym, diff(psi, xSym, false));
//    betas[0][1] = neg(mul(chiSym, diff(psi, ySym, false)));
//
//    // Calculate the symbolic and numeric functions and for the first diagonal using the first "recursion relation"
//    for (int m = 1; m<n-1; m++){
//        int s = m + 1;
//        double c_num = (m + 1.0)/(m + 1.0 + s)*(1 + (s == 1));
//        auto c = mul(real_double(c_num), chiSym);
//        // Symbolic
//        alphas[m][s] = mul(c, sub(diff(alphas[m-1][s-1], xSym, false), diff(betas[m-1][s-1], ySym, false)));
//        betas[m][s] = mul(c, add(diff(betas[m-1][s-1], xSym, false), diff(alphas[m-1][s-1], ySym, false)));
//        // Numeric
//        alphas_lambda[m][s].init({xSym, ySym, gammaSym, chiSym}, *alphas[m][s]);
//        betas_lambda[m][s].init({xSym, ySym, gammaSym, chiSym}, *betas[m][s]);
//    }
//
//    // Calculate the rest of the symbolic and numeric functions using the second "recursion relation"
//    for (int start = 1; start < n; start+=2){
//        for(int s=0; s<(n-start); s++){
//            int m = s + start;
//            double c_num = (m + 1.0)/(m + 1.0 - s)*(1.0 + (s != 0))/2.0;
//            auto c = mul(real_double(c_num), chiSym);
//            // Symbolic
//            alphas[m][s] = mul(c, add(diff(alphas[m-1][s+1], xSym, false), diff(betas[m-1][s+1], ySym, false)));
//            betas[m][s] = mul(c, sub(diff(betas[m-1][s+1], xSym, false), diff(alphas[m-1][s+1], ySym, false)));
//            // Numeric
//            alphas_lambda[m][s].init({xSym, ySym, gammaSym, chiSym}, *alphas[m][s]);
//            betas_lambda[m][s].init({xSym, ySym, gammaSym, chiSym}, *betas[m][s]);
//
//        }
//    }
}
