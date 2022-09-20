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
    drawParallel(imgActual, actualX, actualY);
    cv::Mat imgApparent;

    // if point
    if (mode == 0){
        imgApparent = cv::Mat(size, 2*size, CV_8UC1, cv::Scalar(0, 0, 0));
        imgDistorted = cv::Mat(imgApparent.size(), CV_8UC1, cv::Scalar(0, 0, 0));
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
        drawParallel(imgApparent, apparentX, apparentY);
        parallelDistort(imgApparent, imgDistorted);
    }

    // Copy both the actual and the distorted images into a new matDst array
    cv::Mat matDst(cv::Size(2*size, size), imgActual.type(), cv::Scalar::all(255));
    cv::Mat matRoi = matDst(cv::Rect(0, 0, size, size));
    imgActual.copyTo(matRoi);
    matRoi = matDst(cv::Rect(size, 0, size, size));
    imgDistorted.copyTo(matRoi);

    // Show the matDst array (i.e. both images) in the GUI window.
    cv::imshow("GL Simulator", matDst);

    // Calculate run time for this function and print diagnostic output
    auto endTime = std::chrono::system_clock::now();
    std::cout << "Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count() << " milliseconds" << std::endl;

}


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
}


/* Calculate n! (n factorial) */
double factorial_(unsigned int n){
    double a = 1.0;
    for (int i = 2; i <= n; i++){
        a *= i;
    }
    return a;
}


/* Re-calculate co-ordinates using updated parameter settings from the GUI.
 * This is called from the update() method.                                  */
void Simulator::calculate() {

    CHI = CHI_percent/100.0;

    // Actual position in source plane
    actualX = xPosSlider - size / 2.0;
    actualY = yPosSlider - size / 2.0;

    // Absolute values in source plane
    actualAbs = sqrt(actualX * actualX + actualY * actualY); // Actual distance from the origin
    // The two ratioes correspond to two roots of a quadratic equation.
    double ratio1 = 0.5 + sqrt(0.25 + einsteinR*einsteinR/(CHI*CHI*actualAbs*actualAbs));
    double ratio2 = 0.5 - sqrt(0.25 + einsteinR*einsteinR/(CHI*CHI*actualAbs*actualAbs));
    // Each ratio gives rise to one apparent galaxy.
    apparentAbs = actualAbs*ratio1;
    apparentAbs2 = actualAbs*ratio2;
    // (X,Y) co-ordinates of first image
    apparentX = actualX*ratio1;
    apparentY = actualY*ratio1;
    // (X,Y) co-ordinates of second image
    apparentX2 = actualX*ratio2;
    apparentY2 = actualY*ratio2;
    // BDN: Is the calculation of apparent positions correct above?

    // Projection of apparent position in lens plane
    // This corresponds to scaling the positions in the source plane by the relative distance.
    R = apparentAbs * CHI;
    X = apparentX * CHI;
    Y = apparentY * CHI;

    // Angle relative to x-axis
    // Because of proportionality, the angle is the same for both images.
    phi = atan2(actualY, actualX);
    // BDN: Is this theta in your notes?

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

    std::string filename("50.txt");
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
            // The following two variables are unused.
            // SymEngine::LambdaRealDoubleVisitor alpha_num, beta_num;
            alphas_l[std::stoi(m)][std::stoi(s)].init({x, y, g, c}, *alpha_sym);
            betas_l[std::stoi(m)][std::stoi(s)].init({x, y, g, c}, *beta_sym);
        }
    }
}

