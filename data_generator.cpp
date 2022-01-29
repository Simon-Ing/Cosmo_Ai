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

int window_size = 600;  // Size of source and lens image
int source_size = window_size / 10;   // size of source "Blob"
int einsteinR = window_size / 10;
int xPos = 0;
int actualPos;
cv::Mat apparentSource;
cv::Mat image;


// **************  GENERATE DATA SETTINGS ****************************
static const bool dataGenMode = true; // true for generating data
int iteration_counter = 0;
int DATAPOINTS_TO_GENERATE = 1000;
std::fstream fout;

// **********************************************************

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

            //            // Print some data when evaluating the point at the origin (for debugging)
            //            if (row == window_size/2 && col == window_size/2){
            //                std::cout << "x:  " << x << " y: " << y << " R: " << R << " r: " << r << " theta: " << theta << " EinsteinR: " << einsteinR << std::endl;
            //            }

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
    // for this to work, make a folder called "cosmo_data" in same folder as the main executable

    filename << "datapoint" << iteration_counter << ".png";
    filename_path << "cosmo_data" << "/" << filename.str();
    iteration_counter++;
    cv::imwrite(filename_path.str(), image);

    // Writes new line in .csv file:
    fout << filename.str() << "," << einsteinR << "," << source_size << "," << xPos << " \n";
    std::cout << filename.str() << " generated and saved on drive" << std::endl;
}

void writeToJson() {
    json j;

    std::vector<uchar> array;
    if (image.isContinuous()) {
        array.assign(image.data, image.data + image.total() * image.channels());
    }
    else {
        for (int i = 0; i < image.rows; ++i) {
            array.insert(array.end(), image.ptr<uchar>(i), image.ptr<uchar>(i) + image.cols * image.channels());
        }
    }

    j["einsteinR"] = einsteinR;
    j["source_size"] = source_size;
    j["actualPos"] = actualPos;
    j["image"] = array;

    std::ostringstream filename_path;
    std::ostringstream filename;
    // for this to work, make a folder called "cosmo_data" in same folder as the main executable

    filename << "datapoint" << iteration_counter << ".json";
    filename_path << "data" << "/" << filename.str();
    iteration_counter++;
    std::ofstream file(filename_path.str());
    file << j;
    std::cout << filename.str() << " generated and saved on drive" << std::endl;
}

int main() {
    // Generate dataset:
    //fout.open("cosmo_data/cosmo_data.csv", fout.trunc | fout.in | fout.out);  // opens .csv file
    //fout << "filename" << "," << "einsteinR" << "," << "source_size" << "," << "xPos" << " \n";  // Writes the first line to .csv file

    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_int_distribution<std::mt19937::result_type> rand_einsteinR(1, window_size * 0.15);
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