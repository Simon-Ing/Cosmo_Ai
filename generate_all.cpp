//
// Created by simon on 03.02.2022.
//

#include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp>
#include <iostream>
#include <thread>
#include <random>
#include <fstream>
#include <string>

using namespace nlohmann;

int size = 0;  // Size of source and lens imgDistorted
unsigned long sigma = size / 10;   // size of source "Blob"
unsigned long einsteinR = size / 10;
unsigned long xPos = 0;
int actualPos;
cv::Mat imgApparent;
cv::Mat imgDistorted;
std::string name;


// **************  GENERATE DATA SETTINGS ****************************
static const bool dataGenMode = true; // true for generating data
int iteration_counter = 0;
int DATAPOINTS_TO_GENERATE = 0;
std::fstream fout;

// **********************************************************

void drawGaussian(cv::Mat& img, int& pos) {
    for (int row = 0; row < size; row++) {
        for (int col = 0; col < size; col++) {
            double x = (1.0 * (col - (pos + size / 2.0))) / sigma;
            double y = (size - 1.0 * (row + size / 2.0)) / sigma;

            auto val = (uchar)std::round(255 * std::exp(-x * x - y * y));
            img.at<uchar>(row, col) = val;
        }
    }
}

// Distort the imgDistorted
void distort(int thread_begin, int thread_end, int R) {
    // Evaluate each point in imgDistorted plane ~ lens plane
    for (int row = thread_begin; row < thread_end; row++) {
        for (int col = 0; col <= imgApparent.cols; col++) {
            // Set coordinate system with origin at x=R
            int y = size / 2 - row;

            // How it should be, but looks weird (alt 1 and 2)
            int x = col - size / 2 - R;

            // Calculate distance and angle of the point evaluated relative to center of lens (origin)
            double r = sqrt(x * x + y * y);
            double theta = atan2(y, x);

            // Point mass lens equation
            double frac = (einsteinR * einsteinR * r) / (r * r + R * R + 2 * r * R * cos(theta));
            double x_ = x + frac * (r / R + cos(theta));
            double y_ = y - frac * sin(theta);

            // Translate to array index
            int row_ = size / 2 - (int)round(y_);
            int col_ = size / 2 + R + (int)round(x_);

            // If (x', y') within source, copy value to imgDistorted
            if (row_ < size && col_ < size && row_ > 0 && col_ >= 0) {
                imgDistorted.at<uchar>(row, col) = imgApparent.at<uchar>(row_, col_);
            }
        }
    }
}

void writeToPngFiles() {
    std::ostringstream filename_path;
    std::ostringstream filename;

    filename << einsteinR << "," << sigma << "," << xPos << ".png";
    filename_path << name + "/images/" + filename.str();
    iteration_counter++;
    cv::imwrite(filename_path.str(), imgDistorted);

}

// Split the imgDistorted into n pieces where n is number of threads available and distort the pieces in parallel
static void parallel(int R) {
    unsigned int num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads_vec;
    for (int k = 0; k < num_threads; k++) {
        unsigned int thread_begin = (imgApparent.rows / num_threads) * k;
        unsigned int thread_end = (imgApparent.rows / num_threads) * (k + 1);
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
    actualPos = (xPos - size / 2);
    int sign = 1 - 2*(actualPos < 0);
    apparentPos1 = (int)(actualPos + sqrt(actualPos * actualPos + 4 * einsteinR * einsteinR)*sign) / 2;
    R = apparentPos1;

    // Make the undistorted imgDistorted at the apparent position by making a black background and add a gaussian light source
    imgApparent = cv::Mat(size, size, CV_8UC1, cv::Scalar(0, 0, 0));
    drawGaussian(imgApparent, apparentPos1);

    // Make black background to draw the distorted imgDistorted to
    imgDistorted = cv::Mat(size, size, CV_8UC1, cv::Scalar(0, 0, 0));

    // Run in parallel
    parallel(R);

    //writeToPngFiles();  //this creates .csv file also
    writeToPngFiles();
}

int main(int argc, char *argv[]) {

    size = 400;
    name = std::string("train");

//    std::cout << DATAPOINTS_TO_GENERATE << " " << size << " " << name << std::endl;

    // Generate dataset:

    for (einsteinR = size * 0.0025; einsteinR <= size * 0.125; einsteinR++) {
        for (sigma = size * 0.0025; sigma <= size * 0.125; sigma++) {
            for (xPos = size * 0.25; xPos <= size * 0.75; xPos++) {
                update(0, nullptr);
            }
        }
        std::cout << einsteinR << std::endl;
    }

//    std::random_device dev;
//    std::mt19937 rng(dev());
//    std::uniform_int_distribution<std::mt19937::result_type> rand_einsteinR(size*0.05, size * 0.3);
//    std::uniform_int_distribution<std::mt19937::result_type> rand_source_size(size*0.05, size * 0.3);
//    std::uniform_int_distribution<std::mt19937::result_type> rand_xSlider(size * 0.3, size * 0.7);

//    for (int i = 0; i < DATAPOINTS_TO_GENERATE; i++) {
//        // Randomizes values for eatch iteration
//        einsteinR = rand_einsteinR(rng);
//        sigma = rand_source_size(rng);
//        xPos = rand_xSlider(rng);
//        update(0, nullptr);
//    }
}