#include <opencv2/opencv.hpp>
#include <iostream>
#include <thread>

int window_size = 501;  // Size of source and lens image
int source_size = 50;   // size of source "Blob"
int einsteinR = 30;
double R;
int xPos = 10;
int yPos = 10;
cv::Mat source;
cv::Mat image;
cv::Mat pointSource;


std::vector<int> pointMass(double R_, double r, double theta) {

    // Split the equation into three parts for simplicity. (eqn. 9 from "Gravitational lensing")
    // Find the point from the source corresponding to the point evaluated
    double frac = (einsteinR * einsteinR * r) / (r * r + R_ * R_ + 2 * r * R_ * cos(theta));
    double x_ = r*cos(theta) + frac * (r / R_ + cos(theta));
    double y_ = r*sin(theta) + frac * (-sin(theta));

    // Translate to array index
    int source_row = window_size - (int)round(y_);
    int source_col = (int)round(x_);

    return {source_row, source_col};
}


double findR() {
    for (int y = window_size -1; y >= 0; y--) {
        for (int x = 0; x < window_size; x++) {

            // calculate distance and angle of the point evaluated relative to center of lens (origin)
            double r = sqrt(x * x + y * y);
            double theta = atan2(y, x);

            std::vector<int> sourcePos = pointMass(r, r, theta);

            // If index within source; check if value is > 0, if so return radius
            if (std::max(sourcePos[0], sourcePos[1]) < window_size && std::min(sourcePos[0], sourcePos[1]) >= 0) {
                if (pointSource.at<uchar>(sourcePos[0], sourcePos[1]) > 0) {
                    return r;
                }
            }
        }
    }
    return 0;
}

void distort( int begin, int end) {
    // Evaluate each point in image plane ~ lens plane
    for (int row = begin; row < end; row++) {
        for (int col = 0; col <= source.cols; col++) {

            // set coordinate system with origin in middle and x right and y up
            int x = col;
            int y = window_size - row;

            // calculate distance and angle of the point evaluated relative to center of lens (origin)
            double r = sqrt(x * x + y * y);
            double theta = atan2(y, x);

            std::vector<int> sourcePos = pointMass(R, r, theta);

            // If index within source , copy value to image
            if (std::max(sourcePos[0], sourcePos[1]) < window_size && std::min(sourcePos[0], sourcePos[1]) >= 0) {
                image.at<uchar>(row, col) = source.at<uchar>(sourcePos[0], sourcePos[1]);
            }
        }
    }
}

// Add som lines to the image for reference
void refLines(){
    for (int i = 0; i < window_size; i++) {
//        source.at<uchar>(window_size/2, i) = 255;
//        source.at<uchar>(i, window_size/2) = 255;
        source.at<uchar>(i, window_size - 1) = 255;
        source.at<uchar>(i, 0) = 255;
        source.at<uchar>(window_size - 1, i) = 255;
        source.at<uchar>(0, i) = 255;
    }
}


static void paralell() {//         Run the point mass function for each pixel in image. (multi thread)
    unsigned int num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads_vec;
    for (int k = 0; k < num_threads; k++) {
        unsigned int thread_begin = (source.rows / num_threads) * k;
        unsigned int thread_end = (source.rows / num_threads) * (k + 1);
        std::thread t(distort, thread_begin, thread_end);
        threads_vec.push_back(std::move(t));
        }
    for (auto& thread : threads_vec) {
        thread.join();
    }
}

static void update(int, void*){

    // Find R by iterating through lens plane, and find the point that the center of the source maps to
    pointSource = cv::Mat(window_size, window_size, CV_8UC1, cv::Scalar(0, 0, 0));
    cv::circle(pointSource, cv::Point(xPos, window_size - yPos), 0, cv::Scalar(254, 254, 254), 5);
    R = findR();

    // Make a source image with black background and a gaussian light source placed at xSource, ySource and radius R and an empty image
    source = cv::Mat(window_size, window_size, CV_8UC1, cv::Scalar(0, 0, 0));
    cv::circle(source, cv::Point(xPos, window_size - yPos), 0, cv::Scalar(254, 254, 254), source_size);
    int kSize = (source_size / 2) * 2 + 1; // make sure kernel size is odd
    cv::GaussianBlur(source, source, cv::Size_<int>(kSize, kSize), source_size);
    image = cv::Mat(window_size, window_size, CV_8UC1, cv::Scalar(0, 0, 0));
    refLines();


    // Run with single thread:
//    point_mass_funct(0, window_size);  //for single thread

    // ..or parallel:
    paralell();

    // Scale, format and show on screen
    cv::resize(source, source, cv::Size_<int>(701, 701));
    cv::resize(image, image, cv::Size_<int>(701, 701));
    cv::Mat matDst(cv::Size(source.cols * 2, source.rows), source.type(), cv::Scalar::all(0));
    cv::Mat matRoi = matDst(cv::Rect(0, 0, source.cols, source.rows));
    source.copyTo(matRoi);
    matRoi = matDst(cv::Rect(source.cols, 0, source.cols, source.rows));
    image.copyTo(matRoi);
    cv::imshow("Window", matDst);

}

int main()
{
    cv::namedWindow("Window", cv::WINDOW_AUTOSIZE);
    cv::createTrackbar("source x pos", "Window", &xPos, window_size, update);
    cv::createTrackbar("source y pos", "Window", &yPos, window_size, update);
    cv::createTrackbar("Einstein Radius", "Window", &einsteinR, 1000, update);
    cv::createTrackbar("Source Radius", "Window", &source_size, window_size, update);
    update(0, nullptr);
    cv::waitKey(0);
    return 0;
}

