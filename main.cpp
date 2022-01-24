#include <opencv2/opencv.hpp>
#include <iostream>
#include <thread>

int window_size = 600;  // Size of source and lens image
int source_size = window_size/20;   // size of source "Blob"
int einsteinR = window_size/20;
int R;
int xPos = window_size/2;
int yPos = window_size/2;
cv::Mat source;
cv::Mat image;


void drawGaussian(cv::Mat& img) {
    for (int row = 0; row < window_size; row++) {
        for (int col = 0; col < window_size; col++) {

            double x = (1.0*(col - xPos)) / source_size;
            double y = (window_size - 1.0*(row + yPos)) / source_size;

            uchar val = 255 * std::exp(-x*x - y*y);
            img.at<uchar>(row, col) = val;
        }
    }
}

// Find the corresponding (X', y') for a given (x, y)
void pointMass(int (&target)[2], double R_, double r, double theta) {
    // Split the equation into three parts for simplicity. (eqn. 9 from "Gravitational lensing")
    // Find the point from the source corresponding to the point evaluated
    double frac = (einsteinR * einsteinR * r) / (r * r + R_ * R_ + 2 * r * R_ * cos(theta));
    double x_ = r * cos(theta) + frac * (r / R_ + cos(theta));
    double y_ = r * sin(theta) - frac * sin(theta);

    // Translate to array index
    target[0] = window_size / 2 - (int)round(y_);
    target[1] = (int)round(x_) + window_size / 2 + R;
}

// Distort the image
void distort( int thread_begin, int thread_end) {
    // Evaluate each point in image plane ~ lens plane
    for (int i = thread_begin; i < thread_end; i++) {
        for (int j = 0; j <= source.cols; j++) {

            // set coordinate system with origin at x=R
            int x = j - window_size/2 - R;
            int y = window_size / 2 - i;

            // calculate distance and angle of the point evaluated relative to center of lens (origin)
            double r = sqrt(x * x + y * y);
            double theta = atan2(y, x);

            int sourcePos[2];

            pointMass(sourcePos, R, r, theta);

            // If index within source , copy value to image
            if (std::max(sourcePos[0], sourcePos[1]) < window_size && std::min(sourcePos[0], sourcePos[1]) >= 0) {
                image.at<uchar>(i, j) = source.at<uchar>(sourcePos[0], sourcePos[1]);
            }
        }
    }
}

// Add som lines to the image for reference
void refLines(){
    for (int i = 0; i < window_size; i++) {
        source.at<uchar>(i, window_size - 1) = 255;
        source.at<uchar>(i, 0) = 255;
        source.at<uchar>(window_size - 1, i) = 255;
        source.at<uchar>(0, i) = 255;
    }
}

// Split the image into n pieces where n is number of threads available and distort the pieces in parallel
static void parallel() {
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

// This function is called each time a slider is updated
static void update(int, void*){

    R = xPos;

    // Make the undistorted image by making a black background and add a gaussian light source
    source = cv::Mat(window_size, window_size, CV_8UC1, cv::Scalar(0, 0, 0));
    drawGaussian(source);
//    refLines();

    // Make black background to draw the distorted image to
    image = cv::Mat(window_size, window_size, CV_8UC1, cv::Scalar(0, 0, 0));

    // Run with single thread:
//    distort(0, window_size);

    // ..or parallel:
    parallel();

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
    // Make the user interface and specify the function to be called when moving the sliders: update()
    cv::namedWindow("Window", cv::WINDOW_AUTOSIZE);
    cv::createTrackbar("source x pos", "Window", &xPos, window_size, update);
    cv::createTrackbar("Einstein Radius", "Window", &einsteinR, window_size/10, update);
    cv::createTrackbar("Source Size", "Window", &source_size, window_size/10, update);
    cv::waitKey(0);
    return 0;
}


//// Find the center of the distorted source NOT CURRENTLY IN USE
//double findR() {
//    // Evaluate each point in image plane ~ lens plane
//    for (int i = 0; i < window_size; i++) {
//        for (int j = 0; j <= source.cols; j++) {
//
//            // set coordinate system with origin in middle and x right and y up
//            int x = j - window_size / 2;
//            int y = window_size / 2 - i;
//
//            // calculate distance and angle of the point evaluated relative to center of lens (origin)
//            double r = sqrt(x * x + y * y);
//            double theta = atan2(y, x);
//
//            // Calculate the corresponding (x', y')
//            std::vector<int> sourcePos = pointMass(r, r, theta);
//
//            // If index within source, check if value is > 0, if so we have found R
//            if (std::max(sourcePos[0], sourcePos[1]) < window_size && std::min(sourcePos[0], sourcePos[1]) >= 0) {
//                if (pointSource.at<uchar>(sourcePos[0], sourcePos[1]) > 100) {
//                    return r;
//                }
//            }
//        }
//    }
//    return 0;
//}


// Some questionable solutions to R:

// Find R by iterating through lens plane, and find the point that the center of the source maps to
//    pointSource = cv::Mat(window_size, window_size, CV_8UC1, cv::Scalar(0, 0, 0));
//    cv::circle(pointSource, cv::Point(xPos, window_size - yPos), 0, cv::Scalar(254, 254, 254), 2);
//    R = findR();


//    R = 1.3*einsteinR;


//    R = (xPos - sqrt(xPos*xPos + 4*einsteinR*einsteinR)) / 2;