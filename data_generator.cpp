//
// Created by simon on 29.01.2022.
//

#include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp>
#include <iostream>
#include <thread>
#include <random>
#include <fstream>
#include <string>

using namespace nlohmann;

int window_size = 0;  // Size of source and lens image
unsigned long source_size = window_size / 10;   // size of source "Blob"
unsigned long einsteinR = window_size / 10;
unsigned long xPos = 0;
int actualPos;
cv::Mat apparentSource;
cv::Mat image;
std::string name;


// **************  GENERATE DATA SETTINGS ****************************
static const bool dataGenMode = true; // true for generating data
int iteration_counter = 0;
int DATAPOINTS_TO_GENERATE = 0;
std::fstream fout;

// **********************************************************

void drawGaussian(cv::Mat& img, int& pos) {
    for (int row = 0; row < window_size; row++) {
        for (int col = 0; col < window_size; col++) {
            double x = (1.0 * (col - (pos + window_size / 2.0))) / source_size;
            double y = (window_size - 1.0 * (row + window_size / 2.0)) / source_size;

            auto val = (uchar)std::round(255 * std::exp(-x * x - y * y));
            img.at<uchar>(row, col) = val;
        }
    }
}

// Distort the image
void distort(int thread_begin, int thread_end, int R) {
    // Evaluate each point in image plane ~ lens plane
    for (int row = thread_begin; row < thread_end; row++) {
        for (int col = 0; col <= apparentSource.cols; col++) {
            // Set coordinate system with origin at x=R
            int y = window_size / 2 - row;

            // How it should be, but looks weird (alt 1 and 2)
            int x = col - window_size / 2 - R;

            // Calculate distance and angle of the point evaluated relative to center of lens (origin)
            double r = sqrt(x * x + y * y);
            double theta = atan2(y, x);

            // Point mass lens equation
            double frac = (einsteinR * einsteinR * r) / (r * r + R * R + 2 * r * R * cos(theta));
            double x_ = x + frac * (r / R + cos(theta));
            double y_ = y - frac * sin(theta);

            // Translate to array index
            int row_ = window_size / 2 - (int)round(y_);
            int col_ = window_size / 2 + R + (int)round(x_);

            // If (x', y') within source, copy value to image
            if (row_ < window_size && col_ < window_size && row_ > 0 && col_ >= 0) {
                image.at<uchar>(row, col) = apparentSource.at<uchar>(row_, col_);
            }
        }
    }
}

void writeToPngFiles() {
    std::ostringstream filename_path;
    std::ostringstream filename;

    filename << einsteinR << "," << source_size << "," << xPos << ".png";
    filename_path << name + "/images/" + filename.str();
    iteration_counter++;
    cv::imwrite(filename_path.str(), image);

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
static void update(int, void*) {
    int apparentPos1, apparentPos2, R;
    actualPos = (xPos - window_size / 2);
    int sign = 1 - 2*(actualPos < 0);
    apparentPos1 = (int)(actualPos + sqrt(actualPos * actualPos + 4 * einsteinR * einsteinR)*sign) / 2;
    R = apparentPos1;

    // Make the undistorted image at the apparent position by making a black background and add a gaussian light source
    apparentSource = cv::Mat(window_size, window_size, CV_8UC1, cv::Scalar(0, 0, 0));
    drawGaussian(apparentSource, apparentPos1);

    // Make black background to draw the distorted image to
    image = cv::Mat(window_size, window_size, CV_8UC1, cv::Scalar(0, 0, 0));

    // Run in parallel
    parallel(R);

    //writeToPngFiles();  //this creates .csv file also
    writeToPngFiles();
}

int main(int argc, char *argv[]) {

    DATAPOINTS_TO_GENERATE = atoi(argv[1]);
    window_size = atoi(argv[2]);
    name = std::string(argv[3]);

//    std::cout << DATAPOINTS_TO_GENERATE << " " << window_size << " " << name << std::endl;

    // Generate dataset:

    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_int_distribution<std::mt19937::result_type> rand_einsteinR(window_size * 0.03, window_size * 0.15);
    std::uniform_int_distribution<std::mt19937::result_type> rand_source_size(1, window_size * 0.1);
    std::uniform_int_distribution<std::mt19937::result_type> rand_xSlider(window_size * 0.2, window_size * 0.8);

    for (int i = 0; i < DATAPOINTS_TO_GENERATE; i++) {
        // Randomizes values for eatch iteration
        einsteinR = rand_einsteinR(rng);
        source_size = rand_source_size(rng);
        xPos = rand_xSlider(rng);
        update(0, nullptr);
    }
}