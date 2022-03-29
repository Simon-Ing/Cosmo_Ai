//
// Created by simon on 14.02.2022.
//

#include <opencv2/opencv.hpp>
#include <iostream>
#include <thread>
#include <random>
#include <string>
#define PI 3.14159265358979323846


int size = 0;  // Size of source and lens imgDistorted
unsigned int sigma = size / 10;   // size of source "Blob"
unsigned int einsteinR = size / 10;
unsigned int xPosSlider = 0;
unsigned int yPosSlider = 0;
unsigned int KL_percent = 0;
std::string name;


void drawSource(int begin, int end, cv::Mat& img, int xPos, int yPos) {
    for (int row = begin; row < end; row++) {
        for (int col = 0; col < img.cols; col++) {
            int x = col - xPos - img.cols/2;
            int y = row + yPos - img.rows/2;
            auto value = (uchar)round(255 * exp((-x * x - y * y) / (2.0*sigma*sigma)));
            img.at<uchar>(row, col) = value;
        }
    }
}

void drawParallel(cv::Mat& img, int xPos, int yPos){
    unsigned int num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads_vec;
    for (int k = 0; k < num_threads; k++) {
        unsigned int thread_begin = (img.rows / num_threads) * k;
        unsigned int thread_end = (img.rows / num_threads) * (k + 1);
        std::thread t(drawSource, thread_begin, thread_end, std::ref(img), xPos, yPos);
        threads_vec.push_back(std::move(t));
    }
    for (auto& thread : threads_vec) {
        thread.join();
    }
}

// Distort the image
void distort(int begin, int end, int R, int apparentPos, cv::Mat imgApparent, cv::Mat& imgDistorted, double KL) {
    // Evaluate each point in imgDistorted plane ~ lens plane
    for (int row = begin; row < end; row++) {
        for (int col = 0; col <= imgDistorted.cols; col++) {

            // Set coordinate system with origin at x=R
            int x = col - R - imgDistorted.cols/2;
            int y = imgDistorted.rows/2 - row;

            // Calculate distance and angle of the point evaluated relative to center of lens (origin)
            double r = sqrt(x*x + y*y);
            double theta = atan2(y, x);

            // Point mass lens equation
            double frac = (einsteinR * einsteinR * r) / (r * r + R * R + 2 * r * R * cos(theta));
            double x_ = (x + frac * (r / R + cos(theta))) / KL;
            double y_ = (y - frac * sin(theta)) / KL;

            // Translate to array index
            int row_ = imgApparent.rows / 2 - (int)round(y_);
            int col_ = apparentPos + imgApparent.cols/2 + (int)round(x_);


            // If (x', y') within source, copy value to imgDistorted
            if (row_ < imgApparent.rows && col_ < imgApparent.cols && row_ >= 0 && col_ >= 0) {
                imgDistorted.at<uchar>(row, col) = imgApparent.at<uchar>(row_, col_);
            }
        }
    }
}

// Split the image into (number of threads available) pieces and distort the pieces in parallel
static void parallel(int R, int apparentPos, cv::Mat& imgApparent, cv::Mat& imgDistorted, double KL) {
    unsigned int num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads_vec;
    for (int k = 0; k < num_threads; k++) {
        unsigned int thread_begin = (imgDistorted.rows / num_threads) * k;
        unsigned int thread_end = (imgDistorted.rows / num_threads) * (k + 1);
        std::thread t(distort, thread_begin, thread_end, R, apparentPos, imgApparent, std::ref(imgDistorted), KL);
        threads_vec.push_back(std::move(t));
    }
    for (auto& thread : threads_vec) {
        thread.join();
    }
}

void writeToPngFiles(cv::Mat& image) {
    std::ostringstream filename_path;
    std::ostringstream filename;

    filename  << einsteinR << "," << sigma << "," << xPosSlider << "," << yPosSlider << ".png";
    filename_path << name + "/images/" + filename.str();
    cv::imwrite(filename_path.str(), image);
//    cv::imshow(filename_path.str(), image);
//    cv::waitKey(0);
}

// This function is called each time a slider is updated
static void update(int, void*) {

    //set lower bound on lens distance
    double KL = KL_percent/100.0;

    int xPos = (int)xPosSlider - size/2;
    int yPos = (int)yPosSlider - size/2;
    double phi = atan2(yPos, xPos);

    int actualPos = (int)round(sqrt(xPos*xPos + yPos*yPos));
    int sizeAtLens = (int)round(KL*size);
    int apparentPos = (int)round((actualPos + sqrt(actualPos*actualPos + 4 / (KL*KL) * einsteinR*einsteinR)) / 2.0);
    int R = (int)round(apparentPos * KL);

    // make an image with light source at APPARENT position, make it oversized in width to avoid "cutoff"
    cv::Mat imgApparent(size, 2*size, CV_8UC1, cv::Scalar(0, 0, 0));
    drawParallel(imgApparent, apparentPos, 0);

    // Make empty matrix to draw the distorted image to
    cv::Mat imgDistorted(sizeAtLens, 2*sizeAtLens, CV_8UC1, cv::Scalar(0, 0, 0));

    // Run distortion in parallel
    parallel(R, apparentPos, imgApparent, imgDistorted, KL);

    // make a scaled, rotated and cropped version of the distorted image
    cv::Mat imgDistortedProcessed;
    cv::resize(imgDistorted, imgDistortedProcessed, cv::Size(2 * size, size));
    cv::Mat rot = cv::getRotationMatrix2D(cv::Point(size, size/2), phi*180/PI, 1);
    cv::warpAffine(imgDistortedProcessed, imgDistortedProcessed, rot, cv::Size(2 * size, size));
    imgDistortedProcessed =  imgDistortedProcessed(cv::Rect(size / 2, 0, size, size));
    writeToPngFiles(imgDistortedProcessed);

}

int main(int, char *argv[]) {

    int DATAPOINTS_TO_GENERATE = atoi(argv[1]);
    size = atoi(argv[2]);
    name = std::string(argv[3]);

    // Generate dataset:
    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_int_distribution<std::mt19937::result_type> rand_lens_dist(30, 100);
    std::uniform_int_distribution<std::mt19937::result_type> rand_einsteinR(1, size/10);
    std::uniform_int_distribution<std::mt19937::result_type> rand_source_size(1, size/10);
    std::uniform_int_distribution<std::mt19937::result_type> rand_xSlider(0, size);
    std::uniform_int_distribution<std::mt19937::result_type> rand_ySlider(0, size);

    std::vector<std::vector<unsigned int>> parameters;
    for (int i = 0; i < DATAPOINTS_TO_GENERATE; i++) {
        // Randomizes values for eatch iteration
        KL_percent = 50; //rand_lens_dist(rng);
        einsteinR = rand_einsteinR(rng);
        sigma = rand_source_size(rng);
        xPosSlider = rand_xSlider(rng);
        yPosSlider = rand_xSlider(rng);
        std::vector<unsigned int> params = {KL_percent, einsteinR, sigma, xPosSlider, yPosSlider};

        if ( !std::count(parameters.begin(), parameters.end(), params) ) {
            update(0, nullptr);
            parameters.push_back({KL_percent, einsteinR, sigma, xPosSlider, yPosSlider});
        }
        else{
            i--;
        }
        if (parameters.size() % (DATAPOINTS_TO_GENERATE/10) == 0){
            std::cout << " Datapoints generated: " << parameters.size() << std::endl;
        }
    }
}
