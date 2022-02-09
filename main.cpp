#include <opencv2/opencv.hpp>
#include <thread>
#include <random>
#include <string>
#define _USE_MATH_DEFINES // for C++
#include <math.h>

int size = 400;
int sourceSize = size / 10;
int einsteinR = size / 10;
int xPosSlider = size/2;
int yPosSlider = size/2;
int KL_percent = 50;

// Add some lines to the image for reference
void refLines(cv::Mat& target) {
    int size_ = target.rows;
    for (int i = 0; i < size_; i++) {
        target.at<uchar>(i, size_ / 2) = 150;
        target.at<uchar>(size_ / 2 - 1, i) = 150;
        target.at<uchar>(i, size_ - 1) = 255;
        target.at<uchar>(i, 0) = 255;
        target.at<uchar>(size_ - 1, i) = 255;
        target.at<uchar>(0, i) = 255;
    }
}


void drawSource(cv::Mat& img, int xPos, int yPos) {
	for (int row = 0; row < img.rows; row++) {
		for (int col = 0; col < img.cols; col++) {
			double x = 1.0 * (col - xPos - img.cols/2.0) / sourceSize;
			double y = (size - 1.0 * (row + yPos + size / 2.0)) / sourceSize;
			img.at<uchar>(row, col) = (uchar)round(255 * exp(-x * x - y * y));
		}
	}
}


// Distort the image
void distort(int begin, int end, int R, int apparentPos, cv::Mat imgApparent, cv::Mat& imgDistorted, double KL) {
	// Evaluate each point in imgDistorted plane ~ lens plane
	for (int row = begin; row < end; row++) {
		for (int col = 0; col <= imgDistorted.cols; col++) {

			// Set coordinate system with origin at x=R
            int x = col - R - imgDistorted.cols/2;
			int y = imgDistorted.rows / 2 - row;

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

//// Split the image into (number of threads available) pieces and distort the pieces in parallel
//static void parallel(int R, int apparentPos, cv::Mat& imgApparent, cv::Mat& imgDistorted) {
//	unsigned int num_threads = std::thread::hardware_concurrency();
//	std::vector<std::thread> threads_vec;
//	for (int k = 0; k < num_threads; k++) {
//		unsigned int thread_begin = (imgApparent.rows / num_threads) * k;
//		unsigned int thread_end = (imgApparent.rows / num_threads) * (k + 1);
//		std::thread t(distort, thread_begin, thread_end, R, apparentPos, imgApparent, std::ref(imgDistorted));
//		threads_vec.push_back(std::move(t));
//	}
//	for (auto& thread : threads_vec) {
//		thread.join();
//	}
//}

// This function is called each time a slider is updated
static void update(int, void*) {
    int sign;
    int xPos = xPosSlider - size/2;
    int yPos = yPosSlider - size/2;

    double phi = atan2(yPos, xPos);

    int actualPos = (int)round(sqrt(xPos*xPos + yPos*yPos));
    double KL = std::max(KL_percent/100.0, 0.01);
    int sizeAtLens = (int)round(KL*size);
    if (actualPos == 0) { sign = 1; }
    else{ sign = actualPos / abs(actualPos); }
	int apparentPos = (int)round((actualPos + sqrt(actualPos*actualPos + 4 / (KL*KL) * einsteinR*einsteinR)*sign) / 2.0);
    int apparentPos2 = (int)round((actualPos - sqrt(actualPos*actualPos + 4 / (KL*KL) * einsteinR*einsteinR)*sign) / 2.0);
    int R = (int)round(apparentPos * KL);

	// make an image with light source at APPARENT position, make it oversized in width to avoid "cutoff"
	cv::Mat imgApparent(size, 2*size, CV_8UC1, cv::Scalar(0, 0, 0));
    drawSource(imgApparent, apparentPos, 0);

	// Make empty matrix to draw the distorted image to
	cv::Mat imgDistorted(sizeAtLens, sizeAtLens, CV_8UC1, cv::Scalar(0, 0, 0));

    // Run distortion in parallel
//	parallel(R, apparentPos, imgApparent, imgDistorted);

    distort(0, sizeAtLens, R, apparentPos, imgApparent, imgDistorted, KL);

    //Make a scaled version of the distorted image
    cv::Mat imgDistortedResized;
    cv::resize(imgDistorted, imgDistortedResized, cv::Size(size, size));

    // Make a sliced version of apparent to display
    auto imgApparentDisplay = imgApparent(cv::Rect(size/2,0,size,size));

    cv::Mat imgDistortedResizedRotated;
    cv::Mat rot = cv::getRotationMatrix2D(cv::Point(size/2, size/2), phi*180/3.145, 1);
    cv::warpAffine(imgDistortedResized, imgDistortedResizedRotated, rot, cv::Size(size, size));

    int actualX = (int)round(actualPos*cos(phi));
    int actualY = (int)round(actualPos*sin(phi));
    int apparentX = (int)round(apparentPos*cos(phi));
    int apparentY = (int)round(apparentPos*sin(phi));
    int apparentX2 = (int)round(apparentPos2*cos(phi));
    int apparentY2 = (int)round(apparentPos2*sin(phi));

    // make an image with light source at ACTUAL position
    cv::Mat imgActual(size, size, CV_8UC1, cv::Scalar(0, 0, 0));
    drawSource(imgActual, actualX, actualY);

    cv::putText(imgActual, "Actual position", cv::Point(10,30), cv::FONT_HERSHEY_COMPLEX, 1, cv::Scalar::all(255));
    cv::putText(imgApparentDisplay, "Apparent position", cv::Point(10,30), cv::FONT_HERSHEY_COMPLEX, 1, cv::Scalar::all(255));
    cv::putText(imgDistorted, "Distorted projection", cv::Point(10,30*sizeAtLens/size), cv::FONT_HERSHEY_COMPLEX, KL, cv::Scalar::all(255));
    cv::putText(imgDistortedResizedRotated, "Distorted resized", cv::Point(10,30), cv::FONT_HERSHEY_COMPLEX, 1, cv::Scalar::all(255));
    refLines(imgActual);
    refLines(imgApparentDisplay);
    refLines(imgDistorted);
    refLines(imgDistortedResizedRotated);
    cv::circle(imgDistorted, cv::Point(sizeAtLens/2, sizeAtLens/2), einsteinR, cv::Scalar::all(100));
    cv::circle(imgDistortedResizedRotated, cv::Point(size/2, size/2), (int)round(einsteinR/KL), cv::Scalar::all(100));
    cv::circle(imgDistortedResizedRotated, cv::Point(size/2 + apparentX, size/2 - apparentY), size/50, cv::Scalar::all(150), size/200);
    cv::circle(imgDistortedResizedRotated, cv::Point(size/2 + apparentX2, size/2 - apparentY2), size/50, cv::Scalar::all(150), size/200);
    cv::rectangle(imgDistortedResizedRotated, cv::Point(size/2 + actualX - size/50, size/2 - actualY - size/50), cv::Point(size/2 + actualX + size/50, size/2 - actualY + size/50), cv::Scalar::all(150), size/200);

    cv::Mat matDst(cv::Size(4*size, size), imgActual.type(), cv::Scalar::all(255));
    cv::Mat matRoi = matDst(cv::Rect(0, 0, size, size));
    imgActual.copyTo(matRoi);
    matRoi = matDst(cv::Rect(size, 0, size, size));
    imgApparentDisplay.copyTo(matRoi);
    matRoi = matDst(cv::Rect(2*size, size/2 - sizeAtLens/2, sizeAtLens, sizeAtLens));
    imgDistorted.copyTo(matRoi);
    matRoi = matDst(cv::Rect(3*size, 0, size, size));
    imgDistortedResizedRotated.copyTo(matRoi);
    cv::imshow("Window", matDst);
}

int main()
{
    // Make the user interface and specify the function to be called when moving the sliders: update()
    cv::namedWindow("Window", cv::WINDOW_AUTOSIZE);
    cv::createTrackbar("Einstein Radius:", "Window", &einsteinR, size / 4, update);
    cv::createTrackbar("Source Size        :", "Window", &sourceSize, size / 4, update);
    cv::createTrackbar("Lens dist %        :", "Window", &KL_percent, 100, update);
    cv::createTrackbar("x Pos    :", "Window", &xPosSlider, size, update);
    cv::createTrackbar("y Pos    :", "Window", &yPosSlider, size, update);

    bool running = true;
    while (running) {
        int k = cv::waitKey(30);
        if ((cv::getWindowProperty("Window", cv::WND_PROP_AUTOSIZE) == -1) || (k == 27)) {
            running = false;
        }
    }
    cv::destroyAllWindows();
	return 0;
}
