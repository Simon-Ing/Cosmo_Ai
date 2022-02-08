#include <opencv2/opencv.hpp>
#include <thread>
#include <random>
#include <string>

int size = 500;
int sourceSize = size / 10;
int einsteinR = size / 10;
int actualPos = size / 3;
int KL_percent = 50;

// Add some lines to the image for reference
void refLines(cv::Mat& target) {
    for (int i = 0; i < size; i++) {
        target.at<uchar>(i, size / 2) = 150;
        target.at<uchar>(size / 2 - 1, i) = 150;
        target.at<uchar>(i, size - 1) = 255;
        target.at<uchar>(i, 0) = 255;
        target.at<uchar>(size - 1, i) = 255;
        target.at<uchar>(0, i) = 255;
    }
}


void drawSource(cv::Mat& img, int& pos) {
	for (int row = 0; row < img.rows; row++) {
		for (int col = 0; col < img.cols; col++) {
			double x = 1.0 * (col - (pos)) / sourceSize;
			double y = (size - 1.0 * (row + size / 2.0)) / sourceSize;
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
            int x = col - R;
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
			int col_ = apparentPos + (int)round(x_);


			// If (x', y') within source, copy value to imgDistorted
			if (row_ < imgApparent.rows && col_ < imgApparent.cols && row_ >= 0 && col_ >= 0) {
//                std::cout << "row': " << row_ << " col': " << col_ << std::endl;
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

// size
// sourceSize: 0 - size/4
// einsteinR: 0 - size/4
// actualPos: 0 - size
// KL: 0 - 1

    double KL = std::max(KL_percent/100.0, 0.001);
    int sizeAtLens = (int)round(KL*size);
    int apparentPos = (int)round((actualPos + sqrt(actualPos*actualPos + 4 / (KL*KL) * einsteinR*einsteinR)) / 2.0);
    int R = (int)round(apparentPos * KL);

	// make an image with light source at APPARENT position
	cv::Mat imgApparent(size, size, CV_8UC1, cv::Scalar(0, 0, 0));
    drawSource(imgApparent, apparentPos);

	// Make empty matrix to draw the distorted image to
	cv::Mat imgDistorted(sizeAtLens, sizeAtLens, CV_8UC1, cv::Scalar(0, 0, 0));

    // Run distortion in parallel
//	parallel(R, apparentPos, imgApparent, imgDistorted);

    distort(0, sizeAtLens, R, apparentPos, imgApparent, imgDistorted, KL);

    // make an image with light source at ACTUAL position
	cv::Mat imgActual(size, size, CV_8UC1, cv::Scalar(0, 0, 0));
    drawSource(imgActual, actualPos);

//    // Add some lines for reference, a circle showing einstein radius, a circle at apparent pos and a rectangle at actual pos
//    refLines(imgActual);
//    refLines(imgDistorted);
//    cv::circle(imgDistorted, cv::Point(size / 2, size / 2), einsteinR, 100, size / 400);
//    cv::circle(imgDistorted, cv::Point(R + size / 2, size / 2), 10, 100, size / 400);
////    cv::circle(imgDistorted, cv::Point(R2 + size / 2, size / 2), 10, 100, size / 400);
//    cv::rectangle(imgDistorted, cv::Point((int)(actualPos*KL) + size / 2 - 10, size / 2 - 10), cv::Point((int)(actualPos*KL) + size / 2 + 10, size / 2 + 10), 100, size / 400);

    //Scale, format and show on screen
//    int outputSize = 800;
//    cv::resize(imgActual, imgActual, cv::Size_<int>(outputSize, outputSize));
//    cv::resize(imgDistorted, imgDistorted, cv::Size_<int>(outputSize, outputSize));
//    cv::Mat matDst(cv::Size(imgActual.cols * 2, imgActual.rows), imgActual.type(), cv::Scalar::all(0));
//    cv::Mat matRoi = matDst(cv::Rect(0, 0, imgActual.cols, imgActual.rows));
//    imgActual.copyTo(matRoi);
//    matRoi = matDst(cv::Rect(imgActual.cols, 0, imgActual.cols, imgActual.rows));
//    imgDistorted.copyTo(matRoi);
    cv::imshow("Window", imgActual);
    cv::imshow("distorted", imgDistorted);
    cv::imshow("apparent", imgApparent);
    cv::resize(imgDistorted, imgDistorted, cv::Size(size, size));
    cv::imshow("distorted resized", imgDistorted);
//    cv::imshow("apparent", imgApparent);
}
int main()
{
    // Make the user interface and specify the function to be called when moving the sliders: update()
    cv::namedWindow("Window", cv::WINDOW_AUTOSIZE);
    cv::createTrackbar("Einstein Radius:", "Window", &einsteinR, size / 4, update);
    cv::createTrackbar("Source Size        :", "Window", &sourceSize, size / 4, update);
    cv::createTrackbar("Lens dist %        :", "Window", &KL_percent, 100, update);
    cv::createTrackbar("Actual Pos    :", "Window", &actualPos, size, update);

    bool running = true;
    while (running) {
        running = (cv::waitKey(30) != 27);
    }
    cv::destroyAllWindows();
	return 0;
}
