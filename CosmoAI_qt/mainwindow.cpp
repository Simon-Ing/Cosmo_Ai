#include "mainwindow.h"
#include "./ui_mainwindow.h"
#include <random>
#include <string>
#include <iostream>
#define _USE_MATH_DEFINES // for C++
#include <math.h>
#include <cmath>
#include <QtCore>
#include <QString>
#include <QPainter>
#include <QDebug>
#define PI 3.14159265358979323846

bool grid = true;
bool markers = true;
int wSize = 600;
int einsteinR = wSize/20;
int srcSize = wSize/20;
int KL_percent = 50;
int xPosSlider = wSize/2;
int yPosSlider = wSize/2;

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    imgActual = QImage(wSize, wSize, QImage::Format_RGB32);
    imgApparent = QImage(wSize, wSize, QImage::Format_RGB32);
    imgDistorted = QImage(wSize, wSize, QImage::Format_RGB32);

    // Timer to update values and image
    Timer = new QTimer(this);
    Timer->start(10);
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

//    ui->einsteinSlider->setMinimum(1);
//    ui->einsteinSpinbox->setMinimum(1);

    // Set initial values for UI elements
    ui->einsteinSpinbox->setValue(einsteinR);
    ui->einsteinSlider->setSliderPosition(einsteinR);
    ui->srcSizeSpinbox->setValue(srcSize);
    ui->srcSizeSlider->setSliderPosition(srcSize);
    ui->lensDistSpinbox->setValue(KL_percent);
    ui->lensDistSlider->setSliderPosition(KL_percent);
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

//void MainWindow::refLines(cv::Mat& target){
//    int size_ = target.rows;
//    for (int i = 0; i < size_; i++) {
//        target.at<cv::Vec3b>(i, size_ / 2) = {60, 60, 60};
//        target.at<cv::Vec3b>(size_ / 2 - 1, i) = {60, 60, 60};
//        target.at<cv::Vec3b>(i, size_ - 1) = {255, 255, 255};
//        target.at<cv::Vec3b>(i, 0) = {255, 255, 255};
//        target.at<cv::Vec3b>(size_ - 1, i) = {255, 255, 255};
//        target.at<cv::Vec3b>(0, i) = {255, 255, 255};
//    }
//}

void MainWindow::drawSource(int begin, int end, QImage& img, double xPos, double yPos) {
    for (int row = begin; row < end; row++) {
        for (int col = begin; col < end; col++) {
            double x = col - xPos - end/2.0;
            double y = row + yPos - end/2.0;
            auto val = (uchar)round(255 * exp((-x * x - y * y) / (2.0*srcSize*srcSize)));
            img.setPixel(col, row, qRgb(val, val, val));
        }
    }
}

void MainWindow::distort(int begin, int end, double R, double apparentPos, QImage imgApparent, QImage& imgDistorted, double KL) {
    // Evaluate each point in imgDistorted plane ~ lens plane
    for (int row = begin; row < end; row++) {
        for (int col = begin; col <= end; col++) {

            // Set coordinate system with origin at x=R
            double x = (col - apparentPos - end/2.0) * KL;
            double y = (end/2.0 - row) * KL;

            // Calculate distance and angle of the point evaluated relative to center of lens (origin)
            double r = sqrt(x*x + y*y);
            double theta = atan2(y, x);

            // Point mass lens equation
            double frac = (einsteinR * einsteinR * r) / (r * r + R * R + 2 * r * R * cos(theta));
            double x_ = (x + frac * (r / R + cos(theta))) / KL;
            double y_ = (y - frac * sin(theta)) / KL;

            // Translate to array index
            int row_ = (int)round(end / 2.0 - y_);
            int col_ = (int)round(apparentPos + end/2.0 + x_);


            // If (x', y') within source, copy value to imgDistorted
            if (row_ < wSize && col_ < wSize && row_ > 0 && col_ >= 0 && row < wSize && col < wSize && row > 0 && col >= 0) {
                imgDistorted.setPixel(col, row, imgApparent.pixel(col_, row_));
            }
        }
    }
}

void MainWindow::updateImg() {
    imgApparent.fill(Qt::black);
    imgActual.fill(Qt::black);
    imgDistorted.fill(Qt::black);

    KL_percent = std::max(KL_percent, 30);
    double KL = KL_percent/100.0;
    ui->lensDistSlider->setValue(KL_percent);
    double xPos = xPosSlider - wSize/2.0;
    double yPos = yPosSlider - wSize/2.0;
    double phi = atan2(yPos, xPos);

    double actualPos = sqrt(xPos*xPos + yPos*yPos);
    double apparentPos = (actualPos + sqrt(actualPos*actualPos + 4 / (KL*KL) * einsteinR*einsteinR)) / 2.0;
    double apparentPos2 = (int)round((actualPos - sqrt(actualPos*actualPos + 4 / (KL*KL) * einsteinR*einsteinR)) / 2.0);
    double R = apparentPos * KL;

    drawSource(0, wSize, imgApparent, apparentPos, 0);

    distort(0, wSize, R, apparentPos, imgApparent, imgDistorted, KL);

    // Rotatation of pixmap
    QPixmap pix = QPixmap::fromImage(imgDistorted);
    QPixmap distRot(pix.size());
    QSize pixSize = pix.size();
    distRot.fill(QColor::fromRgb(0, 0, 0, 0));
    QPainter painter(&distRot);
    painter.setRenderHint(QPainter::SmoothPixmapTransform);
    painter.translate(pixSize.height()/2, pixSize.height()/2);
    painter.rotate(-phi*180/PI);
    painter.translate(-pixSize.height()/2, -pixSize.height()/2);
    painter.drawPixmap(0,0, pix);

    int actualX = (int)round(actualPos*cos(phi));
    int actualY = (int)round(actualPos*sin(phi));
    int apparentX = (int)round(apparentPos*cos(phi));
    int apparentY = (int)round(apparentPos*sin(phi));
    int apparentX2 = (int)round(apparentPos2*cos(phi));
    int apparentY2 = (int)round(apparentPos2*sin(phi));

    // make an image with light source at ACTUAL position
    imgActual = QImage(wSize, wSize, QImage::Format_RGB32);
    drawSource(0, wSize, imgActual, actualX, actualY);

    // Draw pixmaps on QLabels
    ui->actLabel->setPixmap(QPixmap::fromImage(imgActual));
    ui->distLabel->setPixmap(distRot);
}

void MainWindow::updateValues() {
    // Set variables to current spinbox values
    einsteinR = ui->einsteinSpinbox->value();
    srcSize = ui->srcSizeSpinbox->value();
    KL_percent = ui->lensDistSpinbox->value();
    xPosSlider = ui->xSpinbox->value();
    yPosSlider = ui->ySpinbox->value();
    grid = ui->gridBox->isChecked();
    markers = ui->markerBox->isChecked();
}

MainWindow::~MainWindow()
{
    delete ui;
}
