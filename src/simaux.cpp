/* (C) 2022: Hans Georg Schaathun <hg@schaathun.net> */

#include "Simulator.h"

// Add some lines to the image for reference
void refLines(cv::Mat& target) {
    int rsize = target.rows;
    int csize = target.cols;
    std::cout << "refLines " << rsize << "x" << csize << "\n" ;
    for (int i = 0; i < rsize ; i++) {
        target.at<cv::Vec3b>(i, csize / 2) = {60, 60, 60};
        target.at<cv::Vec3b>(i, csize - 1) = {255, 255, 255};
        target.at<cv::Vec3b>(i, 0) = {255, 255, 255};
    }
    for (int i = 0; i < csize ; i++) {
        target.at<cv::Vec3b>(rsize / 2 - 1, i) = {60, 60, 60};
        target.at<cv::Vec3b>(rsize - 1, i) = {255, 255, 255};
        target.at<cv::Vec3b>(0, i) = {255, 255, 255};
    }
}

