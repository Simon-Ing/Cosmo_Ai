#include <opencv2/opencv.hpp>
#include <iostream>
#include <thread>

int window_size = 501;  // Size of source and lens image
int source_size = 50;   // size of source "Blob"
int einsteinR = 30;
double R;
int xSlider = window_size / 2; // raw values
int ySlider = window_size / 2;
int xSource;                    // Scaled values (to fit coordinate system)
int ySource;
double K_l = 2;    // Distance to lens
double K_s = 3;    // Distance to source
cv::Mat source;
cv::Mat image;


void point_mass_funct( int thread_begin, int thread_end) {
    // Evaluate each point in image plane ~ lens plane
    for (int i = thread_begin; i < thread_end; i++) {
        for (int j = 0; j <= source.cols; j++) {

            // set coordinate system with origin in middle and x right and y up
            int x = j - window_size / 2;
            int y = window_size / 2 - i;

            // calculate distance and angle of the point evaluated relative to center of lens (origin)
            double r = sqrt(x * x + y * y);
            double theta = atan2(y, x);

            // Split the equation into three parts for simplicity. (eqn. 9 from "Gravitational lensing")
            // Find the point from the source corresponding to the point evaluated
            double frac = (einsteinR * einsteinR * r) / (r * r + R * R + 2 * r * R * cos(theta));
            double x_ = K_s/K_l * (x + frac * (r / R + cos(theta)));
            double y_ = K_s/K_l * (y + frac * (-sin(theta)));

            // Translate to array index
            int source_row = window_size / 2 - (int)round(y_);
            int source_col = (int)round(x_) + window_size / 2;

            // If index within source , copy value to image
            if (std::max(source_col, source_row) < window_size && std::min(source_col, source_row) >= 0) {
                image.at<uchar>(i, j) = source.at<uchar>(source_row, source_col);
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


static void update(int, void*){
    // Scale position of source to fit coordinate system and calculate distance from center of lens(at origin)
    xSource = xSlider - window_size / 2;
    ySource = window_size / 2 - ySlider;
    R = sqrt(xSource*xSource + ySource*ySource);
    std::cout << "x: " << xSource << " y: " << ySource << " R: " << R << std::endl;

    // Make a source image with black background and a gaussian light source placed at xSource, ySource and radius R and an empty image
    source = cv::Mat(window_size, window_size, CV_8UC1, cv::Scalar(0, 0, 0));
    cv::circle(source, cv::Point(xSlider, window_size - ySlider), 0, cv::Scalar(254, 254, 254), source_size);
    int kSize = (source_size / 2) * 2 + 1; // make sure kernel size is odd
    cv::GaussianBlur(source, source, cv::Size_<int>(kSize, kSize), source_size);
    image = cv::Mat(window_size, window_size, CV_8UC1, cv::Scalar(0, 0, 0));
    refLines();

    // Run the point mass function for each pixel in image. (multi thread)
    unsigned int num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads_vec;
    for (int k = 0; k < num_threads; k++) {
        unsigned int thread_begin = (source.rows / num_threads) * k;
        unsigned int thread_end = (source.rows / num_threads) * (k + 1);
        std::thread t(point_mass_funct, thread_begin, thread_end);
        threads_vec.push_back(std::move(t));
        }
    for (auto& thread : threads_vec) {
        thread.join();
    }

    // comment out the stuff above and uncomment the line below to run on single thread
//    point_mass_funct(0, window_size);  //for single thread


    // Some formatting of the images before plotting them
//    source = source(cv::Rect(100,100,source.cols - 200, source.rows - 200));
//    image = image(cv::Rect(100,100,image.cols - 200, image.rows - 200));

    cv::resize(source, source, cv::Size_<int>(701, 701));
    cv::resize(image, image, cv::Size_<int>(701, 701));

    cv::Mat matDst(cv::Size(source.cols*2,source.rows),source.type(),cv::Scalar::all(0));
    cv::Mat matRoi = matDst(cv::Rect(0,0,source.cols,source.rows));
    source.copyTo(matRoi);
    matRoi = matDst(cv::Rect(source.cols,0,source.cols,source.rows));
    image.copyTo(matRoi);

    cv::imshow("Window", matDst);
}

int main()
{
    cv::namedWindow("Window", cv::WINDOW_AUTOSIZE);
    cv::createTrackbar("source x pos", "Window", &xSlider, window_size, update);
    cv::createTrackbar("source y pos", "Window", &ySlider, window_size, update);
    cv::createTrackbar("Einstein Radius", "Window", &einsteinR, 100, update);
    cv::createTrackbar("Source Radius", "Window", &source_size, window_size, update);
    update(0, nullptr);
    cv::waitKey(0);
    return 0;
}
