#include "cosmogui.h"
#include "./ui_cosmogui.h"
#include <opencv2/opencv.hpp>
#include <thread>
#include <random>
#include <string>
#define _USE_MATH_DEFINES // for C++
#include <math.h>
#include <QtCore>

bool grid = true;
bool markers = true;
int wSize = 600;
int einsteinR = wSize/20;
int srcSize = wSize/20;
int lensDist = 50;
int xPosSlider = wSize/2;
int yPosSlider = wSize/2;
int sigma = srcSize;

CosmoGUI::CosmoGUI(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::CosmoGUI)
{
    ui->setupUi(this);

    // Timer to update values and image
    Timer = new QTimer(this);
    Timer->start(20);
    connect(Timer, SIGNAL(timeout()), this, SLOT(updateValues()));
    connect(Timer, SIGNAL(timeout()), this, SLOT(updateImg()));

    // Set max/min values for UI elements
    ui->einsteinSlider->setMaximum(wSize/4);
    ui->einsteinSpinbox->setMaximum(wSize/4);
    ui->srcSizeSlider->setMaximum(wSize/4);
    ui->srcSizeSpinbox->setMaximum(wSize/4);
    ui->lensDistSlider->setMaximum(100);
    ui->lensDistSpinbox->setMaximum(100);
    ui->xSlider->setMaximum(wSize);
    ui->xSpinbox->setMaximum(wSize);
    ui->ySlider->setMaximum(wSize);
    ui->ySpinbox->setMaximum(wSize);

    // Set initial values for UI elements
    ui->einsteinSpinbox->setValue(einsteinR);
    ui->einsteinSlider->setSliderPosition(einsteinR);
    ui->srcSizeSpinbox->setValue(sigma);
    ui->srcSizeSlider->setSliderPosition(sigma);
    ui->lensDistSpinbox->setValue(lensDist);
    ui->lensDistSlider->setSliderPosition(lensDist);
    ui->xSpinbox->setValue(xPosSlider);
    ui->xSlider->setSliderPosition(xPosSlider);
    ui->ySpinbox->setValue(yPosSlider);
    ui->ySlider->setSliderPosition(yPosSlider);
    ui->gridBox->setChecked(true);
    ui->markerBox->setChecked(true);


    // Connect sliders and spinboxes
    connect(ui->einsteinSlider, SIGNAL(valueChanged(int)), ui->einsteinSpinbox, SLOT(setValue(int)));
    connect(ui->einsteinSpinbox, SIGNAL(valueChanged(int)), ui->einsteinSlider, SLOT(setValue(int)));
    connect(ui->srcSizeSpinbox, SIGNAL(valueChanged(int)), ui->srcSizeSlider, SLOT(setValue(int)));
    connect(ui->srcSizeSlider, SIGNAL(valueChanged(int)), ui->srcSizeSpinbox, SLOT(setValue(int)));
    connect(ui->lensDistSpinbox, SIGNAL(valueChanged(int)), ui->lensDistSlider, SLOT(setValue(int)));
    connect(ui->lensDistSlider, SIGNAL(valueChanged(int)), ui->lensDistSpinbox, SLOT(setValue(int)));
    connect(ui->xSpinbox, SIGNAL(valueChanged(int)), ui->xSlider, SLOT(setValue(int)));
    connect(ui->xSlider, SIGNAL(valueChanged(int)), ui->xSpinbox, SLOT(setValue(int)));
    connect(ui->ySpinbox, SIGNAL(valueChanged(int)), ui->ySlider, SLOT(setValue(int)));
    connect(ui->ySlider, SIGNAL(valueChanged(int)), ui->ySpinbox, SLOT(setValue(int)));
}

void CosmoGUI::refLines(cv::Mat& target){
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

void CosmoGUI::drawSource(cv::Mat& img, int xPos, int yPos) {
    for (int row = 0; row < img.rows; row++) {
        for (int col = 0; col < img.cols; col++) {
            int x = col - xPos - img.cols/2;
            int y = row + yPos - img.rows/2;
            auto value = (uchar)round(255 * exp((-x * x - y * y) / (2.0*sigma*sigma)));
            img.at<uchar>(row, col) = value;
        }
    }
}

void CosmoGUI::distort(int begin, int end, int R, int apparentPos, cv::Mat imgApparent, cv::Mat& imgDistorted, double KL) {
    // Evaluate each point in imgDistorted plane ~ lens plane
    for (int row = begin; row < end; row++) {
        for (int col = 0; col <= imgDistorted.cols; col++) {

            // Set coordinate system with origin at x=R
            int x = col - R - imgDistorted.cols/2;
            int y = imgDistorted.rows/2 - row;

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

void CosmoGUI::updateImg() {
    int xPos = xPosSlider - wSize/2;
    int yPos = yPosSlider - wSize/2;
    double phi = atan2(yPos, xPos);

    int actualPos = (int)round(sqrt(xPos*xPos + yPos*yPos));
    double KL = std::max(lensDist/100.0, 0.01);
    int sizeAtLens = (int)round(KL*wSize);
    int apparentPos = (int)round((actualPos + sqrt(actualPos*actualPos + 4 / (KL*KL) * einsteinR*einsteinR)) / 2.0);
    int apparentPos2 = (int)round((actualPos - sqrt(actualPos*actualPos + 4 / (KL*KL) * einsteinR*einsteinR)) / 2.0);
    int R = (int)round(apparentPos * KL);

    // make an image with light source at APPARENT position, make it oversized in width to avoid "cutoff"
    cv::Mat imgApparent(wSize, 2*wSize, CV_8UC1, cv::Scalar(0, 0, 0));
    drawSource(imgApparent, apparentPos, 0);

    // Make empty matrix to draw the distorted image to
    cv::Mat imgDistorted(sizeAtLens, 2*sizeAtLens, CV_8UC1, cv::Scalar(0, 0, 0));

    // Run distortion in parallel

    distort(0, sizeAtLens, R, apparentPos, imgApparent, imgDistorted, KL);

    // make a scaled, rotated and cropped version of the distorted image
    cv::Mat imgDistortedDisplay;
    cv::resize(imgDistorted, imgDistortedDisplay, cv::Size(2*wSize, wSize));
    cv::Mat rot = cv::getRotationMatrix2D(cv::Point(wSize, wSize/2), phi*180/3.145, 1);
    cv::warpAffine(imgDistortedDisplay, imgDistortedDisplay, rot, cv::Size(2*wSize, wSize));
    imgDistortedDisplay =  imgDistortedDisplay(cv::Rect(wSize/2, 0, wSize, wSize));
    cv::cvtColor(imgDistortedDisplay, imgDistortedDisplay, cv::COLOR_GRAY2BGR);

    int actualX = (int)round(actualPos*cos(phi));
    int actualY = (int)round(actualPos*sin(phi));
    int apparentX = (int)round(apparentPos*cos(phi));
    int apparentY = (int)round(apparentPos*sin(phi));
    int apparentX2 = (int)round(apparentPos2*cos(phi));
    int apparentY2 = (int)round(apparentPos2*sin(phi));

    // make an image with light source at ACTUAL position
    cv::Mat imgActual(wSize, wSize, CV_8UC1, cv::Scalar(0, 0, 0));
    drawSource(imgActual, actualX, actualY);

    cv::cvtColor(imgActual, imgActual, cv::COLOR_GRAY2BGR);

    int displaySize = 600;

    if (grid == true) {
        refLines(imgActual);
        refLines(imgDistortedDisplay);
    }

    if (markers == true) {
        cv::circle(imgDistortedDisplay, cv::Point(wSize/2, wSize/2), (int)round(einsteinR/KL), cv::Scalar::all(60));
        cv::drawMarker(imgDistortedDisplay, cv::Point(wSize/2 + apparentX, wSize/2 - apparentY), cv::Scalar(0, 0, 255), cv::MARKER_TILTED_CROSS, displaySize/30);
        cv::drawMarker(imgDistortedDisplay, cv::Point(wSize/2 + apparentX2, wSize/2 - apparentY2), cv::Scalar(0, 0, 255), cv::MARKER_TILTED_CROSS, displaySize/30);
        cv::drawMarker(imgDistortedDisplay, cv::Point(wSize/2 + actualX, wSize/2 - actualY), cv::Scalar(255, 0, 0), cv::MARKER_TILTED_CROSS, displaySize/30);
    }

    cv::resize(imgActual, imgActual, cv::Size(displaySize, displaySize));
    cv::resize(imgDistortedDisplay, imgDistortedDisplay, cv::Size(displaySize, displaySize));


    cv::Mat matDst(cv::Size(2*displaySize, displaySize), imgActual.type(), cv::Scalar::all(255));
    cv::Mat matRoi = matDst(cv::Rect(0, 0, displaySize, displaySize));
    imgActual.copyTo(matRoi);
    matRoi = matDst(cv::Rect(displaySize, 0, displaySize, displaySize));
    imgDistortedDisplay.copyTo(matRoi);

    // Convert opencv Mat to QImage and display on label
    QImage imdisplay((uchar*)matDst.data, matDst.cols, matDst.rows, matDst.step, QImage::Format_RGB888);
    ui->imgLabel->setPixmap(QPixmap::fromImage(imdisplay));

}

void CosmoGUI::updateValues() {
    // Set variables to current spinbox values
    einsteinR = ui->einsteinSpinbox->value();
    sigma = ui->srcSizeSpinbox->value();
    lensDist = ui->lensDistSpinbox->value();
    xPosSlider = ui->xSpinbox->value();
    yPosSlider = ui->ySpinbox->value();
    grid = ui->gridBox->isChecked();
    markers = ui->markerBox->isChecked();
}

CosmoGUI::~CosmoGUI()
{
    delete ui;
}
