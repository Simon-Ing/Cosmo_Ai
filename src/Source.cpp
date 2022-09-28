/* (C) 2022: Hans Georg Schaathun <hg@schaathun.net> */

#include "Simulator.h"
#include <symengine/expression.h>
#include <symengine/lambda_double.h>
#include <thread>
#include <symengine/parser.h>
#include <fstream>

double factorial_(unsigned int n);

Source::Source(int sz,double sig) :
        size(sz),
        sigma(sig)
{ 
    // imgActual = cv::Mat(size, size, CV_8UC1, cv::Scalar(0, 0, 0));
    imgApparent = cv::Mat(size, size, CV_8UC1, cv::Scalar(0, 0, 0));
    drawParallel( imgApparent ) ;
}

/* Getters for the images */
// cv::Mat Source::getActual() { return imgActual ; }
cv::Mat Source::getImage() { return imgApparent ; }

/* drawParallel() split the image into chunks to draw it in parallel using drawSource() */
void Source::drawParallel(cv::Mat& dst){
    int n_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads_vec;
    for (int i = 0; i < n_threads; i++) {
        int begin = dst.rows / n_threads * i;
        int end = dst.rows / n_threads * (i + 1);
        std::thread t([begin, end, &dst, this]() { drawSource(begin, end, dst ); });
        threads_vec.push_back(std::move(t));
    }
    for (auto& thread : threads_vec) {
        thread.join();
    }
}


/* Draw the source image.  The sourceSize is interpreted as the standard deviation in a Gaussian distribution */
void Source::drawSource(int begin, int end, cv::Mat& dst) {
    for (int row = begin; row < end; row++) {
        for (int col = 0; col < dst.cols; col++) {
            int x = col - dst.cols/2;
            int y = row - dst.rows/2;
            auto value = (uchar)round(255 * exp((-x * x - y * y) / (2.0*sigma*sigma)));
            dst.at<uchar>(row, col) = value;
        }
    }
}

