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
int xPos = 0;
int yPos = 0;

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    imgActual = QImage(wSize, wSize, QImage::Format_RGB32);
    imgApparent = QImage(1.5*wSize, wSize, QImage::Format_RGB32);
    imgDistorted = QImage(1.5*wSize, wSize, QImage::Format_RGB32);

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
    ui->xSlider->setMaximum(wSize/2);
    ui->xSpinbox->setMaximum(wSize/2);
    ui->xSlider->setMinimum(-wSize/2);
    ui->xSpinbox->setMinimum(-wSize/2);
    ui->ySlider->setMaximum(wSize/2);
    ui->ySpinbox->setMaximum(wSize/2);
    ui->ySlider->setMinimum(-wSize/2);
    ui->ySpinbox->setMinimum(-wSize/2);

//    ui->einsteinSlider->setMinimum(1);
//    ui->einsteinSpinbox->setMinimum(1);

    // Set initial values for UI elements
    ui->einsteinSpinbox->setValue(einsteinR);
    ui->einsteinSlider->setSliderPosition(einsteinR);
    ui->srcSizeSpinbox->setValue(srcSize);
    ui->srcSizeSlider->setSliderPosition(srcSize);
    ui->lensDistSpinbox->setValue(KL_percent);
    ui->lensDistSlider->setSliderPosition(KL_percent);
    ui->xSpinbox->setValue(xPos);
    ui->xSlider->setSliderPosition(xPos);
    ui->ySpinbox->setValue(yPos);
    ui->ySlider->setSliderPosition(yPos);
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

void MainWindow::drawSource(QImage& img, double xPos, double yPos) {
    int rows = img.height();
    int cols = img.width();
    for (int row = 0; row < rows; row++) {
        for (int col = 0; col < cols; col++) {
            double x = col - xPos - cols/2;
            double y = -yPos - row + rows/2;
            auto val = (uchar)round(255 * exp((-x * x - y * y) / (2.0*srcSize*srcSize)));
            img.setPixel(col, row, qRgb(val, val, val));
        }
    }
}

void MainWindow::distort(QImage imgApparent, QImage& imgDistorted, double R, double apparentPos, double KL) {
    int rows = imgDistorted.height();
    int cols = imgDistorted.width();
    // Evaluate each point in imgDistorted plane ~ lens plane
    for (int row = 0; row < rows; row++) {
        for (int col = 0; col <= cols; col++) { // <= ???????????????????????????????????????

            // Set coordinate system with origin at x=R
            double x = (col - apparentPos - cols/2.0) * KL;
            double y = (rows/2.0 - row) * KL;

            // Calculate distance and angle of the point evaluated relative to center of lens (origin)
            double r = sqrt(x*x + y*y);
            double theta = atan2(y, x);

            // Point mass lens equation
            double frac = (einsteinR * einsteinR * r) / (r * r + R * R + 2 * r * R * cos(theta));
            double x_ = (x + frac * (r / R + cos(theta))) / KL;
            double y_ = (y - frac * sin(theta)) / KL;

            // Translate to array index
            int row_ = (int)round(rows/2.0 - y_);
            int col_ = (int)round(apparentPos + cols/2.0 + x_);


            // If (x', y') within source, copy value to imgDistorted
            if (row_ < rows && col_ < cols && row_ > 0 && col_ >= 0) {
            imgDistorted.setPixel(col, row, imgApparent.pixel(col_, row_));
            }
        }
    }
}

void MainWindow::updateImg() {
    imgApparent.fill(Qt::black);
    imgActual.fill(Qt::black);
    imgDistorted.fill(Qt::black);

    double KL = KL_percent/100.0;
    double phi = atan2(yPos, xPos);

    double actualPos = sqrt(xPos*xPos + yPos*yPos);
    double apparentPos = (actualPos + sqrt(actualPos*actualPos + 4 / (KL*KL) * einsteinR*einsteinR)) / 2.0;
    double apparentPos2 = (int)round((actualPos - sqrt(actualPos*actualPos + 4 / (KL*KL) * einsteinR*einsteinR)) / 2.0);
    double R = apparentPos * KL;

    drawSource(imgApparent, apparentPos, 0);

    distort(imgApparent, imgDistorted, R, apparentPos, KL);

    // Rotatation of pixmap
    QPixmap pix = QPixmap::fromImage(imgDistorted);
    QPixmap distRot(pix.size());
    QSize pixSize = pix.size();

    distRot.fill(QColor::fromRgb(Qt::black));
    QPainter painter(&distRot);
    painter.setRenderHint(QPainter::SmoothPixmapTransform);
    painter.translate(pixSize.width()/2, pixSize.height()/2);
    painter.rotate(-phi*180/PI);
    painter.translate(-pixSize.width()/2, -pixSize.height()/2);
    painter.drawPixmap(0,0, pix);

    // Crop rotated pixmap to correct display size
    QRect rect(wSize/4, 0, wSize, wSize);
    QPixmap distRotCrop = distRot.copy(rect);

//    QString sizeString = QString("(%1,%2)").arg(distRot2.width()).arg(distRot2.height());
//    qDebug(qUtf8Printable(sizeString));

    int actualX = (int)round(actualPos*cos(phi));
    int actualY = (int)round(actualPos*sin(phi));
    int apparentX = (int)round(apparentPos*cos(phi));
    int apparentY = (int)round(apparentPos*sin(phi));
    int apparentX2 = (int)round(apparentPos2*cos(phi));
    int apparentY2 = (int)round(apparentPos2*sin(phi));

    if (grid == true) {
        QPainter painter(&distRotCrop);
        QPen pen(Qt::gray, 2, Qt::DashLine);
        painter.setPen(pen);
        painter.setOpacity(0.3);

        QLineF lineVert(wSize/2, 0, wSize/2, wSize);
        QLineF lineHor(0, wSize/2, wSize, wSize/2);
        painter.drawLine(lineVert);
        painter.drawLine(lineHor);

        QPointF center(wSize/2, wSize/2);
        painter.drawEllipse(center, (int)round(einsteinR/KL), (int)round(einsteinR/KL));
    }

    if (markers == true) {
        QPointF apparentPoint1(wSize/2 + apparentX, wSize/2 - apparentY);
        QPointF apparentPoint2(wSize/2 + apparentX2, wSize/2 - apparentY2);
        QPointF actualPoint(wSize/2 + actualX, wSize/2 - actualY);

        QPainter painter(&distRotCrop);
        QPen bluePen(Qt::blue, 10);
        QPen redPen(Qt::red, 10);
        painter.setPen(bluePen);
        painter.setOpacity(0.4);
        painter.drawPoint(apparentPoint1);
        painter.drawPoint(apparentPoint2);
        painter.setPen(redPen);
        painter.drawPoint(actualPoint);
    }

    // make an image with light source at ACTUAL position
    imgActual = QImage(wSize, wSize, QImage::Format_RGB32);
    drawSource(imgActual, actualX, actualY);

    // Draw pixmaps on QLabels
    ui->actLabel->setPixmap(QPixmap::fromImage(imgActual));
    ui->distLabel->setPixmap(distRotCrop);
}

void MainWindow::updateValues() {
    // Set variables to current spinbox values
    einsteinR = ui->einsteinSpinbox->value();
    srcSize = ui->srcSizeSpinbox->value();
    KL_percent = ui->lensDistSpinbox->value();
    xPos = ui->xSpinbox->value();
    yPos = ui->ySpinbox->value();
    grid = ui->gridBox->isChecked();
    markers = ui->markerBox->isChecked();
}

MainWindow::~MainWindow()
{
    delete ui;
}
