#include "mainwindow.h"
#include "./ui_mainwindow.h"
#include <string>
#include <iostream>
#include <QtCore>
#include <QString>
#include <QPainter>
#include <QDebug>
#include <QPainterPath>
#include <QInputDialog>
#include <QStyleFactory>
#define PI 3.14159265358979323846



MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    this->setWindowTitle("CosmoAI");
    init_values();
    setup();
    theme();
    MainWindow::adjustSize();
}


void MainWindow::setup(){
    imgActual = QImage(wSize, wSize, QImage::Format_RGB32);
    imgApparent = QImage(2*wSize, wSize, QImage::Format_RGB32);
    imgDistorted = QImage(2*wSize, wSize, QImage::Format_RGB32);
    rocket = QPixmap(":/images/images/Tintin.png");

    // Set max/min values for UI elements
    ui->einsteinSlider->setMaximum(0.1*wSize);
    ui->einsteinSpinbox->setMaximum(0.1*wSize);
    ui->srcSizeSlider->setMaximum(0.1*wSize);
    ui->srcSizeSpinbox->setMaximum(0.1*wSize);
    ui->lensDistSlider->setMaximum(100);
    ui->lensDistSpinbox->setMaximum(100);
    ui->lensDistSlider->setMinimum(30);
    ui->lensDistSpinbox->setMinimum(30);
    ui->xSlider->setMaximum(wSize/2);
    ui->xSpinbox->setMaximum(wSize/2);
    ui->xSlider->setMinimum(-wSize/2);
    ui->xSpinbox->setMinimum(-wSize/2);
    ui->ySlider->setMaximum(wSize/2);
    ui->ySpinbox->setMaximum(wSize/2);
    ui->ySlider->setMinimum(-wSize/2);
    ui->ySpinbox->setMinimum(-wSize/2);

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

void MainWindow::init_values() {
    grid = true;
    markers = true;
    legendCheck = true;
    gridSize = 2;
    einsteinR = wSize/20;
    srcSize = wSize/20;
    KL = 0.65;
    actualX = 0;
    actualY = 0;
    source = ui->srcTypeComboBox->currentText();

    // Set initial values for UI elements
    ui->einsteinSpinbox->setValue(einsteinR);
    ui->einsteinSlider->setSliderPosition(einsteinR);
    ui->srcSizeSpinbox->setValue(srcSize);
    ui->srcSizeSlider->setSliderPosition(srcSize);
    ui->lensDistSpinbox->setValue(KL*100);
    ui->lensDistSlider->setSliderPosition(KL*100);
    ui->xSpinbox->setValue(actualX);
    ui->xSlider->setSliderPosition(actualX);
    ui->ySpinbox->setValue(actualY);
    ui->ySlider->setSliderPosition(actualY);
    ui->gridBox->setChecked(grid);
    ui->markerBox->setChecked(markers);
    ui->actionMarkers->setChecked(markers);
    ui->actionLegend->setChecked(legendCheck);
}

void MainWindow::drawGaussian(int begin, int end, QImage& img, double xPos, double yPos) {
    int rows = img.height();
    int cols = img.width();
    for (int row = begin; row < end; row++) {
        for (int col = 0; col < cols; col++) {
            double x = col - xPos - cols/2;
            double y = -yPos - row + rows/2;
            int val = round(255 * exp((-x * x - y * y) / (2.0*srcSize*srcSize)));
            img.setPixel(col, row, qRgb(val, val, val));
        }
    }
}

void MainWindow::theme(){
    if (darkMode) {
        qApp->setStyle(QStyleFactory::create("Fusion"));
        QPalette darkPalette;
        darkPalette.setColor(QPalette::Window,QColor(53,53,53));
        darkPalette.setColor(QPalette::WindowText,Qt::white);
        darkPalette.setColor(QPalette::Disabled,QPalette::WindowText,QColor(127,127,127));
        darkPalette.setColor(QPalette::Base,QColor(42,42,42));
        darkPalette.setColor(QPalette::AlternateBase,QColor(66,66,66));
        darkPalette.setColor(QPalette::ToolTipBase,Qt::white);
        darkPalette.setColor(QPalette::ToolTipText,Qt::white);
        darkPalette.setColor(QPalette::Text,Qt::white);
        darkPalette.setColor(QPalette::Disabled,QPalette::Text,QColor(127,127,127));
        darkPalette.setColor(QPalette::Dark,QColor(35,35,35));
        darkPalette.setColor(QPalette::Shadow,QColor(20,20,20));
        darkPalette.setColor(QPalette::Button,QColor(53,53,53));
        darkPalette.setColor(QPalette::ButtonText,Qt::white);
        darkPalette.setColor(QPalette::Disabled,QPalette::ButtonText,QColor(127,127,127));
        darkPalette.setColor(QPalette::BrightText,Qt::red);
        darkPalette.setColor(QPalette::Link,QColor(42,130,218));
        darkPalette.setColor(QPalette::Highlight,QColor(42,130,218));
        darkPalette.setColor(QPalette::Disabled,QPalette::Highlight,QColor(80,80,80));
        darkPalette.setColor(QPalette::HighlightedText,Qt::white);
        darkPalette.setColor(QPalette::Disabled,QPalette::HighlightedText,QColor(127,127,127));
        qApp->setPalette(darkPalette);

    } else{
        qApp->setPalette(this->style()->standardPalette());
    }
}

void MainWindow::drawSource(){
    if (source == "Gauss"){
        drawGaussianThreaded(imgActual, actualX, actualY);
//        std::cout << "actualX: " << actualX << " actualY: " << actualY << " apparentX: " << apparentX << " apparentY: " << apparentY << std::endl;
        drawGaussianThreaded(imgApparent, apparentX, apparentY);
    }
    else{
        QPainter pAct(&imgActual);
        QPainter pApp(&imgApparent);
        QPen pen(Qt::white, wSize/200);
        pAct.setPen(pen);
        pApp.setPen(pen);

        if (source == "Rocket"){
            QPixmap rocket1 = rocket.scaled(6*srcSize, 6*srcSize, Qt::KeepAspectRatio, Qt::SmoothTransformation);
            QPoint posApp(apparentX + wSize - rocket1.width()/2, -apparentY + wSize/2 - rocket1.height()/2);
            QPoint posAct(actualX + wSize/2 - rocket1.width()/2, wSize/2 - actualY - rocket1.height()/2);
            pAct.drawPixmap(posAct, rocket1);
            pApp.drawPixmap(posApp, rocket1);
        }
        QRect rectApp(apparentX + wSize - srcSize, -apparentY + wSize/2 - srcSize, 2*srcSize, 2*srcSize);
        QRect rectAct(actualX + wSize/2 - srcSize, wSize/2 - actualY - srcSize, 2*srcSize, 2*srcSize);
        if (source == "Circle"){
            pAct.drawEllipse(rectAct);
            pApp.drawEllipse(rectApp);
        }
        else if (source == "Square"){
            pAct.drawRect(rectAct);
            pApp.drawRect(rectApp);
        }

        else if (source == "Triangle"){
            QPainterPath pathAct;
            pathAct.moveTo(rectAct.left() + (rectAct.width() / 2), rectAct.top());
            pathAct.lineTo(rectAct.bottomLeft());
            pathAct.lineTo(rectAct.bottomRight());
            pathAct.lineTo(rectAct.left() + (rectAct.width() / 2), rectAct.top());
            pAct.fillPath(pathAct, QBrush(Qt::white));

            QPainterPath pathApp;
            pathApp.moveTo(rectApp.left() + (rectApp.width() / 2), rectApp.top());
            pathApp.lineTo(rectApp.bottomLeft());
            pathApp.lineTo(rectApp.bottomRight());
            pathApp.lineTo(rectApp.left() + (rectApp.width() / 2), rectApp.top());
            pApp.fillPath(pathApp, QBrush(Qt::white));
        }
    }
}

void MainWindow::distort(int begin, int end) {
    int rows = imgDistorted.height();
    int cols = imgDistorted.width();
    // Evaluate each point in imgDistorted plane ~ lens plane
    for (int row = begin; row < end; row++) {
        for (int col = 0; col <= cols; col++) { // <= ???????????????????????????????????????

            // Set coordinate system with origin at x=R
            double x = (col - apparentAbs - cols/2.0) * KL;
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
            int col_ = (int)round(apparentAbs + cols/2.0 + x_);


            // If (x', y') within source, copy value to imgDistorted
            if (row_ < rows && col_ < cols && row_ > 0 && col_ >= 0) {
            imgDistorted.setPixel(col, row, imgApparent.pixel(col_, row_));
            }
        }
    }
}


void MainWindow::drawGaussianThreaded(QImage& img, double xPos, double yPos){
    unsigned int num_threads = std::thread::hardware_concurrency();
    int tasks = img.height() / num_threads;
    int remainder = img.height() % num_threads;

    std::vector<std::thread> threads_vec;
    for (unsigned int k = 0; k < num_threads; k++) {
        unsigned int thread_begin = tasks * k;
        unsigned int thread_end = tasks * (k + 1);

        if (k == num_threads - 1 && remainder != 0) {
            thread_end = thread_end + remainder;
        }

        std::thread t(&MainWindow::drawGaussian, this, thread_begin, thread_end, std::ref(img), xPos, yPos);
        threads_vec.push_back(std::move(t));
    }
    for (auto& thread : threads_vec) {
        thread.join();
    }
}


// Split the image into (number of threads available) pieces and distort the pieces in parallel
void MainWindow::distortThreaded() {
    unsigned int num_threads = std::thread::hardware_concurrency();
    int tasks = imgDistorted.height() / num_threads;
    int remainder = imgDistorted.height() % num_threads;

    std::vector<std::thread> threads_vec;
    for (unsigned int k = 0; k < num_threads; k++) {
        unsigned int thread_begin = tasks * k;
        unsigned int thread_end = tasks * (k + 1);

        if (k == num_threads - 1 && remainder != 0) {
            thread_end = thread_end + remainder;
        }

        std::thread t(&MainWindow::distort, this, thread_begin, thread_end);
        threads_vec.push_back(std::move(t));
    }
    for (auto& thread : threads_vec) {
        thread.join();
    }
}


QPixmap MainWindow::rotate(QPixmap src, double angle,int x, int y){
    QPixmap r(src.size());
//    QSize s = src.size();
    r.fill(QColor::fromRgb(Qt::black));
    QPainter m(&r);
    m.setRenderHint(QPainter::SmoothPixmapTransform);
    m.translate(src.width()/2 + x, src.height()/2 + y);
    m.rotate(angle*180/PI);
    m.translate(-src.width()/2 - x, -src.height()/2 - y);
    m.drawPixmap(0,0, src);
    return r;
}


void MainWindow::drawRadius(QPixmap& src){
    QPointF center(src.width()/2, src.height()/2);
    QPainter painter(&src);
    QPen pen(Qt::gray, 2, Qt::DashLine);
    painter.setPen(pen);
    painter.setOpacity(0.3);
    painter.drawEllipse(center, (int)round(einsteinR/KL), (int)round(einsteinR/KL));
}


void MainWindow::drawGrid(QPixmap& img){
    QPainter painter(&img);
    QPen pen(Qt::gray, 2, Qt::SolidLine);
    painter.setPen(pen);
    painter.setOpacity(0.3);


    if (gridSize > 0) {
        int remainder = (wSize%gridSize)/2;
        for (int var = wSize/gridSize; var < wSize;) {
            QLineF lineVert(wSize-var-remainder, 0, wSize-var-remainder, wSize-remainder);
            QLineF lineHor(0, wSize-var-remainder, wSize-remainder, wSize-var-remainder);
            painter.drawLine(lineVert);
            painter.drawLine(lineHor);
            var+=wSize/gridSize;
        }
    }
}


void MainWindow::drawMarker(QPixmap& src, int x, int y, int size, QColor color){
    QPointF point(x, y);
    QPainter painter(&src);
    QPen pen(color, size);
    painter.setPen(pen);
    painter.setOpacity(0.4);
    painter.drawPoint(point);
}

void MainWindow::resizeEvent(QResizeEvent *event){
    QMainWindow::resizeEvent(event);
    updateImg();
}

void MainWindow::drawText(QPixmap& img, int x, int y, int fontSize, QString text){
    QPointF point(x,y);
    QPainter painter(&img);
    QFont font;
    font.setPixelSize(fontSize);
    painter.setFont(font);
    painter.drawText(point, text);
}

void MainWindow::drawLegend(QPixmap& img){

    // Create legend pixmap
    int legendHeight = ui->distLabel->height()/12;
    int legendWidth = ui->distLabel->height()/4;
    QPixmap legend(legendWidth, legendHeight);

    // Background color of legend
    legend.fill(Qt::gray);

    // Draw markers in legend
    int markerSize = legendWidth/16;
    drawMarker(legend, legendWidth/10, legendHeight/2 - markerSize, markerSize, Qt::red);
    drawMarker(legend, legendWidth/10, legendHeight/2 + markerSize, markerSize, Qt::blue);

    int fontSize = legendWidth/12;
    drawText(legend, legendWidth/15 + markerSize*2, legendHeight/2 - markerSize + markerSize/2, fontSize, "Actual position");
    drawText(legend, legendWidth/15 + markerSize*2, legendHeight/2 + markerSize + markerSize/2, fontSize, "Apparent positions");

    // Set legend opacity and draw to main pixmap
    QPainter painter(&img);
    int xPos = 0;
    if (apparentX < -wSize/6 && apparentY > wSize/6){
        xPos += wSize - legendWidth;
    }
    painter.setOpacity(0.6);
    painter.drawPixmap(xPos, 0, legend);
}

void MainWindow::updateImg() {
    if (actualX == 0){
        actualX = 1; // a cheap trick
    }
    // Reset images
    imgApparent.fill(Qt::black);
    imgActual.fill(Qt::black);
    imgDistorted.fill(Qt::black);

    // Calculate positions and angles
    phi = atan2(actualY, actualX);
    actualAbs = sqrt(actualX*actualX + actualY*actualY);
    apparentAbs = (actualAbs + sqrt(actualAbs*actualAbs + 4 / (KL*KL) * einsteinR*einsteinR)) / 2.0;
    apparentAbs2 = (actualAbs - sqrt(actualAbs*actualAbs + 4 / (KL*KL) * einsteinR*einsteinR)) / 2.0;
    double shiftRatio = apparentAbs / actualAbs;
    double shiftRatio2 = apparentAbs2 / actualAbs;
    apparentX = actualX * shiftRatio;
    apparentY = actualY * shiftRatio;
    apparentX2 = actualX * shiftRatio2;
    apparentY2 = actualY * shiftRatio2;
    R = apparentAbs * KL;

    drawSource();

    // Convert image to pixmap
    QPixmap imgAppPix = QPixmap::fromImage(imgApparent);

    // Pre rotate pixmap
    imgAppPix = rotate(imgAppPix, phi, 0, 0);

    // Make a copy to display and crop it
    QRect rect2(wSize/2, 0, wSize, wSize);
    auto imgAppPixDisp = imgAppPix.copy(rect2);

    // Convert pre-rotated pixmap to image and distort the image
    imgApparent = imgAppPix.toImage();
    distortThreaded();

    // Convert distorted image to pixmap, rotate and crop
    QPixmap imgDistPix = QPixmap::fromImage(imgDistorted);
    imgDistPix = rotate(imgDistPix, -phi, 0, 0);
    QRect rect(wSize/2, 0, wSize, wSize);
    imgDistPix = imgDistPix.copy(rect);

    auto imgActPix = QPixmap::fromImage(imgActual);

    // Draw grids and markers
    if (grid == true) {
        drawGrid(imgActPix);
        drawGrid(imgAppPixDisp);
        drawGrid(imgDistPix);
        drawRadius(imgDistPix);
    }
    if (markers) {

        drawMarker(imgDistPix, wSize/2 + apparentX, wSize/2 - apparentY, 10, Qt::blue);
        drawMarker(imgDistPix, wSize/2 + apparentX2, wSize/2 - apparentY2, 10, Qt::blue);
        drawMarker(imgDistPix, wSize/2 + actualX, wSize/2 - actualY, 10, Qt::red);
    }

    // Scale pixmaps to fit in labels
    int labelH = ui->actLabel->height();
    imgDistPix = imgDistPix.scaled(labelH, labelH, Qt::KeepAspectRatio, Qt::SmoothTransformation);
    imgActPix = imgActPix.scaled(labelH, labelH, Qt::KeepAspectRatio, Qt::SmoothTransformation);

    // Draw legend after scaling to ensure text clarity
    if (legendCheck && markers) {
      drawLegend(imgDistPix);
    }

    // Draw pixmaps on QLabels
    ui->actLabel->setPixmap(imgActPix);
    ui->distLabel->setPixmap(imgDistPix);
}


MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::on_einsteinSpinbox_valueChanged()
{
    // Set variables to current spinbox values
    einsteinR = ui->einsteinSpinbox->value();
    updateImg();
}


void MainWindow::on_srcSizeSpinbox_valueChanged()
{
    srcSize = ui->srcSizeSpinbox->value();
    updateImg();
}


void MainWindow::on_lensDistSpinbox_valueChanged(int arg1)
{
    KL = arg1/100.0;
    updateImg();
}


void MainWindow::on_xSpinbox_valueChanged()
{
    actualX = ui->xSpinbox->value();
    updateImg();
}


void MainWindow::on_ySpinbox_valueChanged()
{
    actualY = ui->ySpinbox->value();
    updateImg();
}


void MainWindow::on_gridBox_stateChanged(int arg1)
{
    grid = arg1;
    updateImg();
}


void MainWindow::on_markerBox_stateChanged(int arg1)
{
    markers = arg1;
    ui->actionMarkers->setChecked(arg1);
    updateImg();
}


void MainWindow::on_resetButton_clicked()
{
    init_values();
    updateImg();
}


void MainWindow::on_srcTypeComboBox_currentTextChanged(const QString &arg1)
{
    source = arg1;
    updateImg();
}


void MainWindow::on_actionReset_triggered()
{
    init_values();
    updateImg();
}


void MainWindow::on_actionMarkers_toggled(bool arg1)
{
    markers = arg1;
    ui->markerBox->setChecked(arg1);
}


void MainWindow::on_actionLegend_toggled(bool arg1)
{
    legendCheck = arg1;
    updateImg();
}


void MainWindow::on_actionOff_triggered()
{
    ui->gridBox->setChecked(false);
    updateImg();
}


void MainWindow::on_action2x2_triggered()
{
    gridSize = 2;
    ui->gridBox->setChecked(true);
    updateImg();
}

void MainWindow::on_action4x4_triggered()
{
    gridSize = 4;
    ui->gridBox->setChecked(true);
    updateImg();
}


void MainWindow::on_action8x8_triggered()
{
    gridSize = 8;
    ui->gridBox->setChecked(true);
    updateImg();
}

void MainWindow::on_action12x12_triggered()
{
    gridSize = 12;
    ui->gridBox->setChecked(true);
    updateImg();
}


void MainWindow::on_actionChange_resolution_triggered()
{
    QString currentSize = QString::number(wSize);
    unsigned int input = QInputDialog::getInt(this,"Change resolution", "Current: " + currentSize, wSize, 50, 2048);
    wSize = input;
    setup();
    updateImg();
}


void MainWindow::on_actionCustom_triggered()
{
    QString currentSize = QString::number(gridSize);
    unsigned int input = QInputDialog::getInt(this,"Custom grid size", "Current: " + currentSize + "x" + currentSize, gridSize, 2, 100);
    gridSize = input;
    updateImg();
}


void MainWindow::on_actionDark_mode_toggled(bool arg1)
{
    darkMode = arg1;
    theme();
}

