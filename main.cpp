#include <opencv2/opencv.hpp>
#include <iostream>
#include <thread>

int window_size = 600;  // Size of source and lens image
int source_size = window_size/10;   // size of source "Blob"
int einsteinR = window_size/10;
int xPos = 0;
cv::Mat apparentSource;
cv::Mat image;

void drawGaussian(cv::Mat& img, int& pos) {
    for (int row = 0; row < window_size; row++) {
        for (int col = 0; col < window_size; col++) {

            double x = (1.0*(col - (pos + window_size/2.0))) / source_size;
            double y = (window_size - 1.0*(row + window_size/2.0)) / source_size;

            auto val = (uchar)std::round(255 * std::exp(-x*x - y*y));
            img.at<uchar>(row, col) = val;
        }
    }
}

// Distort the image
void distort( int thread_begin, int thread_end, int R) {
    // Evaluate each point in image plane ~ lens plane
    for (int row = thread_begin; row < thread_end; row++) {
        for (int col = 0; col <= apparentSource.cols; col++) {

            // Set coordinate system with origin at x=R
            int y = window_size/2 - row;

            // How it should be, but looks weird (alt 1 and 2)
            int x = col - xPos;

            // Calculate distance and angle of the point evaluated relative to center of lens (origin)
            double r = sqrt(x * x + y * y);
            double theta = atan2(y, x);


            // Point mass lens equation
            double frac = (einsteinR * einsteinR * r) / (r * r + R * R + 2 * r * R * cos(theta));
            double x_ = x + frac * (r / R + cos(theta));
            double y_ = y - frac * sin(theta);

//            // Print some data when evaluating the point at the origin (for debugging)
//            if (row == window_size/2 && col == window_size/2){
//                std::cout << "x:  " << x << " y: " << y << " R: " << R << " r: " << r << " theta: " << theta << " EinsteinR: " << einsteinR << std::endl;
//            }

            // Translate to array index
            int row_ = window_size/2 - (int)round(y_);
            int col_ = xPos + (int)round(x_);

            // If (x', y') within source, copy value to image
            if (row_ < window_size && col_ < window_size && row_ > 0 && col_ >= 0) {
                image.at<uchar>(row, col) = apparentSource.at<uchar>(row_, col_);
            }
        }
    }
}

// Add some lines to the image for reference
void refLines(cv::Mat& target){
    for (int i = 0; i < window_size; i++) {
        target.at<uchar>(i, window_size/2) = 150;
        target.at<uchar>(window_size/2 - 1, i) = 150;
        target.at<uchar>(i, window_size - 1) = 255;
        target.at<uchar>(i, 0) = 255;
        target.at<uchar>(window_size - 1, i) = 255;
        target.at<uchar>(0, i) = 255;
    }
}

// Split the image into n pieces where n is number of threads available and distort the pieces in parallel
static void parallel(int R) {
    unsigned int num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads_vec;
    for (int k = 0; k < num_threads; k++) {
        unsigned int thread_begin = (apparentSource.rows / num_threads) * k;
        unsigned int thread_end = (apparentSource.rows / num_threads) * (k + 1);
        std::thread t(distort, thread_begin, thread_end, R);
        threads_vec.push_back(std::move(t));
        }
    for (auto& thread : threads_vec) {
        thread.join();
    }
}

// This function is called each time a slider is updated
static void update(int, void*){

    int apparentPos = xPos - window_size/2;
    if (apparentPos == 0) apparentPos = 1;

    int R = apparentPos; // Should be absolute value but that looks weird

    int actualPos = apparentPos - einsteinR*einsteinR/R;

    // Make the undistorted image at the apparent position by making a black background and add a gaussian light source
    apparentSource = cv::Mat(window_size, window_size, CV_8UC1, cv::Scalar(0, 0, 0));
    drawGaussian(apparentSource, apparentPos);

    // Make black background to draw the distorted image to
    image = cv::Mat(window_size, window_size, CV_8UC1, cv::Scalar(0, 0, 0));

    // Run with single thread:
//    distort(0, window_size);

    // ..or parallel:
    parallel(R);

    // Make the undistorted image at the ACTUAL position by making a black background and add a gaussian light source
    cv::Mat actualSource(window_size, window_size, CV_8UC1, cv::Scalar(0, 0, 0));
    drawGaussian(actualSource, actualPos);

    // Add some lines for reference
    refLines(actualSource);
    refLines(image);
    cv::circle(image, cv::Point(window_size/2, window_size/2), einsteinR, 100, window_size/400);
    cv::circle(image, cv::Point(xPos, window_size/2), 10, 100, window_size/400);
    cv::circle(image, cv::Point(actualPos + window_size/2, window_size/2), 10, 100, window_size/400);

    // Scale, format and show on screen
    cv::resize(actualSource, actualSource, cv::Size_<int>(700, 700));
    cv::resize(image, image, cv::Size_<int>(700, 700));
    cv::Mat matDst(cv::Size(actualSource.cols * 2, actualSource.rows), actualSource.type(), cv::Scalar::all(0));
    cv::Mat matRoi = matDst(cv::Rect(0, 0, actualSource.cols, actualSource.rows));
    actualSource.copyTo(matRoi);
    matRoi = matDst(cv::Rect(actualSource.cols, 0, actualSource.cols, actualSource.rows));
    image.copyTo(matRoi);
    cv::imshow("Window", matDst);
}

int main()
{
    // Make the user interface and specify the function to be called when moving the sliders: update()
    cv::namedWindow("Window", cv::WINDOW_AUTOSIZE);
    cv::createTrackbar("source x pos", "Window", &xPos, window_size, update);
    cv::createTrackbar("Einstein Radius", "Window", &einsteinR, window_size/4, update);
    cv::createTrackbar("Source Size", "Window", &source_size, window_size/4, update);
    bool running = true;
    while(running) {
        running = (cv::waitKey(0) != 27);
    }
    return 0;
}