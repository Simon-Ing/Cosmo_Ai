#include <opencv2/opencv.hpp>
#include <thread>
#include <random>
#include <string>
#include <cmath>
#define PI 3.14159265358979323846

int size = 600;
int sigma = size / 20;
int einsteinR = size / 20;
int xPosSlider = size/2;
int yPosSlider = size/2;
int KL_percent = 50;

// Add some lines to the image for reference
void refLines(cv::Mat& target) {
    int size_ = target.rows;
    for (int i = 0; i < size_; i++) {
        target.at<cv::Vec3b>(i, size_ / 2) = {60, 60, 60};
        target.at<cv::Vec3b>(size_ / 2 - 1, i) = {60, 60, 60};
        target.at<cv::Vec3b>(i, size_ - 1) = {255, 255, 255};
        target.at<cv::Vec3b>(i, 0) = {255, 255, 255};
        target.at<cv::Vec3b>(size_ - 1, i) = {255, 255, 255};
        target.at<cv::Vec3b>(0, i) = {255, 255, 255};
    }
}


void drawSource(int begin, int end, cv::Mat& img, double xPos, double yPos) {
	for (int row = begin; row < end; row++) {
		for (int col = 0; col < img.cols; col++) {
			double x = col - xPos - img.cols/2.0;
			double y = row + yPos - img.rows/2.0;
            auto value = (uchar)round(255 * exp((-x * x - y * y) / (2.0*sigma*sigma)));
			img.at<uchar>(row, col) = value;
		}
	}
}

void drawParallel(cv::Mat& img, double xPos, double yPos){
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
void distort(int begin, int end, double R, double apparentPos, cv::Mat imgApparent, cv::Mat& imgDistorted, double KL) {
	// Evaluate each point in imgDistorted plane ~ lens plane
	for (int row = begin; row < end; row++) {
		for (int col = 0; col <= imgDistorted.cols; col++) {

			// Set coordinate system with origin at x=R
            double x = (col - apparentPos - imgDistorted.cols/2.0) * KL;
			double y = (imgDistorted.rows/2.0 - row) * KL;

			// Calculate distance and angle of the point evaluated relative to center of lens (origin)
			double r = sqrt(x*x + y*y);
			double theta = atan2(y, x);

			// Point mass lens equation
			double frac = (einsteinR * einsteinR * r) / (r * r + R * R + 2 * r * R * cos(theta));
			double x_ = (x + frac * (r / R + cos(theta))) / KL;
			double y_ = (y - frac * sin(theta)) / KL;

            // Translate to array index
			int row_ = (int)round(imgApparent.rows / 2.0 - y_);
			int col_ = (int)round(apparentPos + imgApparent.cols/2.0 + x_);


			// If (x', y') within source, copy value to imgDistorted
			if (row_ < imgApparent.rows && col_ < imgApparent.cols && row_ >= 0 && col_ >= 0) {
                imgDistorted.at<uchar>(row, col) = imgApparent.at<uchar>(row_, col_);
			}
		}
	}
}

// Split the image into (number of threads available) pieces and distort the pieces in parallel
static void parallel(double R, double apparentPos, cv::Mat& imgApparent, cv::Mat& imgDistorted, double KL) {
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

// This function is called each time a slider is updated
static void update(int, void*) {

    //set lower bound on lens distance
    KL_percent = std::max(KL_percent, 30);
    double KL = KL_percent/100.0;
    cv::setTrackbarPos("Lens dist %    :", "GL Simulator", KL_percent);

    double xPos = xPosSlider - size/2.0;
    double yPos = yPosSlider - size/2.0;
    double phi = atan2(yPos, xPos);

    double actualPos = sqrt(xPos*xPos + yPos*yPos);
	double apparentPos = (actualPos + sqrt(actualPos*actualPos + 4 / (KL*KL) * einsteinR*einsteinR)) / 2.0;
    double apparentPos2 = (int)round((actualPos - sqrt(actualPos*actualPos + 4 / (KL*KL) * einsteinR*einsteinR)) / 2.0);
    double R = apparentPos * KL;

	// make an image with light source at APPARENT position, make it oversized in width to avoid "cutoff"
	cv::Mat imgApparent(size, 2*size, CV_8UC1, cv::Scalar(0, 0, 0));
    drawParallel(imgApparent, apparentPos, 0);

	// Make empty matrix to draw the distorted image to
	cv::Mat imgDistorted(size, 2*size, CV_8UC1, cv::Scalar(0, 0, 0));

    // Run distortion in parallel
	parallel(R, apparentPos, imgApparent, imgDistorted, KL);

    // make a scaled, rotated and cropped version of the distorted image
    cv::Mat imgDistortedDisplay;
    cv::resize(imgDistorted, imgDistortedDisplay, cv::Size(2*size, size));
    cv::Mat rot = cv::getRotationMatrix2D(cv::Point(size, size/2), phi*180/PI, 1);
    cv::warpAffine(imgDistortedDisplay, imgDistortedDisplay, rot, cv::Size(2*size, size));
    imgDistortedDisplay =  imgDistortedDisplay(cv::Rect(size/2, 0, size, size));
    cv::cvtColor(imgDistortedDisplay, imgDistortedDisplay, cv::COLOR_GRAY2BGR);

    int actualX = (int)round(actualPos*cos(phi));
    int actualY = (int)round(actualPos*sin(phi));
    int apparentX = (int)round(apparentPos*cos(phi));
    int apparentY = (int)round(apparentPos*sin(phi));
    int apparentX2 = (int)round(apparentPos2*cos(phi));
    int apparentY2 = (int)round(apparentPos2*sin(phi));

    // make an image with light source at ACTUAL position
    cv::Mat imgActual(size, size, CV_8UC1, cv::Scalar(0, 0, 0));
    drawParallel(imgActual, actualX, actualY);

    cv::cvtColor(imgActual, imgActual, cv::COLOR_GRAY2BGR);

    int displaySize = 600;

    refLines(imgActual);
    refLines(imgDistortedDisplay);
    cv::circle(imgDistortedDisplay, cv::Point(size/2, size/2), (int)round(einsteinR/KL), cv::Scalar::all(60));
    cv::drawMarker(imgDistortedDisplay, cv::Point(size/2 + apparentX, size/2 - apparentY), cv::Scalar(0, 0, 255), cv::MARKER_TILTED_CROSS, size/30);
    cv::drawMarker(imgDistortedDisplay, cv::Point(size/2 + apparentX2, size/2 - apparentY2), cv::Scalar(0, 0, 255), cv::MARKER_TILTED_CROSS, size/30);
    cv::drawMarker(imgDistortedDisplay, cv::Point(size/2 + actualX, size/2 - actualY), cv::Scalar(255, 0, 0), cv::MARKER_TILTED_CROSS, size/30);
    cv::resize(imgActual, imgActual, cv::Size(displaySize, displaySize));
    cv::resize(imgDistortedDisplay, imgDistortedDisplay, cv::Size(displaySize, displaySize));
    cv::Mat matDst(cv::Size(2*displaySize, displaySize), imgActual.type(), cv::Scalar::all(255));
    cv::Mat matRoi = matDst(cv::Rect(0, 0, displaySize, displaySize));
    imgActual.copyTo(matRoi);
    matRoi = matDst(cv::Rect(displaySize, 0, displaySize, displaySize));
    imgDistortedDisplay.copyTo(matRoi);

    cv::imshow("GL Simulator", matDst);

}

int main()
{
    // Make the user interface and specify the function to be called when moving the sliders: update()
    cv::namedWindow("GL Simulator", cv::WINDOW_AUTOSIZE);
    cv::createTrackbar("Lens dist %    :", "GL Simulator", &KL_percent, 100, update);
    cv::createTrackbar("Einstein radius:", "GL Simulator", &einsteinR, size / 10, update);
    cv::createTrackbar("Source sigma   :", "GL Simulator", &sigma, size / 10, update);
    cv::createTrackbar("X position     :", "GL Simulator", &xPosSlider, size, update);
    cv::createTrackbar("Y position     :", "GL Simulator", &yPosSlider, size, update);

    bool running = true;
    while (running) {
        int k = cv::waitKey(1);
        if ((cv::getWindowProperty("GL Simulator", cv::WND_PROP_AUTOSIZE) == -1) || (k == 27)) {
            running = false;
        }
    }
    cv::destroyAllWindows();
	return 0;
}
