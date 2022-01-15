#include <opencv2/opencv.hpp>
#include <iostream>
#include <thread>


//using namespace std;

int window_size = 3000;
int R_e = 30;
int R = 40;
int x_pos = window_size / 2;
int y_pos = window_size / 2;
double K_l = 1.0;

double K_s = 1.1;
cv::Mat source;
cv::Mat image;


void point_mass_funct( int thread_begin, int thread_end) {
    for (int i = thread_begin; i < thread_end; i++) {
        for (int j = 0; j <= source.cols; j++) {
            int x = j - window_size / 2;
            int y = window_size / 2 - i;
            double r = sqrt(x * x + y * y);
            double theta = atan2(y, x);
            double frac = (R_e * R_e * r) / (r * r + R * R + 2 * r * R * cos(theta));
            double x_ = K_s / K_l * x + frac * (r / R + cos(theta));
            double y_ = K_s / K_l * y + frac * (-sin(theta));
            int source_row = window_size / 2 - (int)y_;
            int source_col = (int)x_ + window_size / 2;

            // If source position within source image
            if (std::max(source_col, source_row) < window_size && std::min(source_col, source_row) >= 0) {
                //mutex here if needed. ..?
                image.at<uchar>(i, j) = source.at<uchar>(source_row, source_col);
            }
        }
    }
}




static void update(int, void*){
    R = std::max(R, 1);
    source = cv::Mat(window_size, window_size, CV_8UC1, cv::Scalar(0, 0, 0));

//    for(int i = -100; i < 101; i += 100){
//        cv::circle(source, cv::Point(x_pos + i, y_pos), 0, cv::Scalar(255, 255, 255), R);
//        cv::circle(source, cv::Point(x_pos, y_pos + i), 0, cv::Scalar(255, 255, 255), R);
//    }
    cv::circle(source, cv::Point(x_pos, y_pos), 0, cv::Scalar(255, 255, 255), R);

    int sigma = (R % 2) ? R : R-1;
    cv::GaussianBlur(source, source, cv::Size_<int>(sigma, sigma), sigma);

    image = cv::Mat(window_size, window_size, CV_8UC1, cv::Scalar(0, 0, 0));

    // This is where the magic happens
    int num_threads = std::thread::hardware_concurrency();
    
    std::vector<std::thread> threads_vec;
    for (int k = 0; k < num_threads; k++) {
        int thread_begin = (source.rows / num_threads) * k;
        int thread_end = (source.rows / num_threads) * k + (source.rows / num_threads);
                   
        std::thread t(point_mass_funct, thread_begin, thread_end);
        threads_vec.push_back(std::move(t));
            
        //point_mass_funct(thread_begin, thread_end);  //for single thread
        }
    for (auto& thread : threads_vec) {
        thread.join();
    }


    source = source(cv::Rect(50,50,source.cols - 100, source.rows - 100));
    image = image(cv::Rect(50,50,image.cols - 100, image.rows - 100));

    cv::resize(source, source, cv::Size_<int>(501, 501));
    cv::resize(image, image, cv::Size_<int>(501, 501));

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
    cv::createTrackbar("x pos", "Window", &x_pos, window_size, update);
    cv::createTrackbar("y pos", "Window", &y_pos, window_size, update);
    cv::createTrackbar("Einstein", "Window", &R_e, 100, update);
    cv::createTrackbar("Radius", "Window", &R, 100, update);
    cv::waitKey(0);
    return 0;
}