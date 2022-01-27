#include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp>
#include <iostream>
#include <thread>
#include <random>
#include <fstream>
#include <string>

using json = nlohmann::json;

int window_size = 600;  // Size of source and lens image
int source_size = window_size / 10;   // size of source "Blob"
int einsteinR = window_size / 10;
int xPos = 0;
int actualPos;
cv::Mat apparentSource;
cv::Mat image;
static const bool actualMode = true; // true for actual pos false for apparent

// **************  GENERATE DATA SETTINGS ****************************
static const bool dataGenMode = true; // true for generating data
int iteration_counter = 0;
int DATAPOINTS_TO_GENERATE = 1000;
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

// Add some lines to the image for reference
void refLines(cv::Mat& target) {
	for (int i = 0; i < window_size; i++) {
		target.at<uchar>(i, window_size / 2) = 150;
		target.at<uchar>(window_size / 2 - 1, i) = 150;
		target.at<uchar>(i, window_size - 1) = 255;
		target.at<uchar>(i, 0) = 255;
		target.at<uchar>(window_size - 1, i) = 255;
		target.at<uchar>(0, i) = 255;
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
	filename_path << "cosmo_data" << "/" << filename.str();
	iteration_counter++;
	std::ofstream file(filename_path.str());
	file << j;
	std::cout << filename.str() << " generated and saved on drive" << std::endl;
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

	if (actualMode) {
		actualPos = (xPos - window_size / 2);
		//        int sign = 1 - 2*(actualPos < 0);
		apparentPos1 = (int)(actualPos + sqrt(actualPos * actualPos + 4 * einsteinR * einsteinR)) / 2;
		apparentPos2 = (int)(actualPos - sqrt(actualPos * actualPos + 4 * einsteinR * einsteinR)) / 2;
		R = apparentPos1;
	}

	else {
		apparentPos1 = xPos - window_size / 2;
		if (apparentPos1 == 0) apparentPos1 = 1;
		R = apparentPos1; // Should be absolute value but that looks weird
		actualPos = apparentPos1 - einsteinR * einsteinR / R;
	}

	// Make the undistorted image at the apparent position by making a black background and add a gaussian light source
	apparentSource = cv::Mat(window_size, window_size, CV_8UC1, cv::Scalar(0, 0, 0));
	drawGaussian(apparentSource, apparentPos1);

	// Make black background to draw the distorted image to
	image = cv::Mat(window_size, window_size, CV_8UC1, cv::Scalar(0, 0, 0));

	// Run with single thread:
//    distort(0, window_size);

	// ..or parallel:
	parallel(R);

	// Make the undistorted image at the ACTUAL position by making a black background and add a gaussian light source
	cv::Mat actualSource(window_size, window_size, CV_8UC1, cv::Scalar(0, 0, 0));
	drawGaussian(actualSource, actualPos);

	if (!dataGenMode) {
		// Add some lines for reference, a circle showing einstein radius, a circle at apparent po and a rectangle at actual pos
		refLines(actualSource);
		refLines(image);
		cv::circle(image, cv::Point(window_size / 2, window_size / 2), einsteinR, 100, window_size / 400);
		cv::circle(image, cv::Point(apparentPos1 + window_size / 2, window_size / 2), 10, 100, window_size / 400);
		if (actualMode) {
			cv::circle(image, cv::Point(apparentPos2 + window_size / 2, window_size / 2), 10, 100, window_size / 400);
		}
		cv::rectangle(image, cv::Point(actualPos + window_size / 2 - 10, window_size / 2 - 10), cv::Point(actualPos + window_size / 2 + 10, window_size / 2 + 10), 100, window_size / 400);

		//Scale, format and show on screen
		int outputSize = 800;
		cv::resize(actualSource, actualSource, cv::Size_<int>(outputSize, outputSize));
		cv::resize(image, image, cv::Size_<int>(outputSize, outputSize));
		cv::Mat matDst(cv::Size(actualSource.cols * 2, actualSource.rows), actualSource.type(), cv::Scalar::all(0));
		cv::Mat matRoi = matDst(cv::Rect(0, 0, actualSource.cols, actualSource.rows));

		actualSource.copyTo(matRoi);
		matRoi = matDst(cv::Rect(actualSource.cols, 0, actualSource.cols, actualSource.rows));
		image.copyTo(matRoi);
		cv::imshow("Window", matDst);
	}
	else {
		//writeToPngFiles();  //this creates .csv file also
		writeToJson();
	}
}
int main()
{
	if (!dataGenMode) {
		// Make the user interface and specify the function to be called when moving the sliders: update()
		cv::namedWindow("Window", cv::WINDOW_AUTOSIZE);
		cv::createTrackbar("Einstein Radius:", "Window", &einsteinR, window_size / 4, update);
		cv::createTrackbar("Source Size        :", "Window", &source_size, window_size / 4, update);

		if (actualMode) {
			cv::createTrackbar("Actual Pos    :", "Window", &xPos, window_size, update);
		}
		else {
			cv::createTrackbar("Apparent Pos   :", "Window", &xPos, window_size, update);
		}
		bool running = true;
		while (running) {
			running = (cv::waitKey(30) != 27);
		}
		cv::destroyAllWindows();
	}
	else {
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

	return 0;
}