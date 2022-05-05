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
#include <QFileDialog>
#include <QMessageBox>
#define PI 3.14159265358979323846


void MainWindow::updateImg() {
    if (actualX == 0){
        actualX = 1; // a cheap trick
    }
    // Reset images
    imgApparent.fill(Qt::black);
    imgActual.fill(Qt::black);
    imgDistorted.fill(Qt::black);

    calculateStuff();

    drawSource();

    // Convert image to pixmap
    imgAppPix = QPixmap::fromImage(imgApparent);
    // Pre rotate pixmap
    imgAppPix = rotate(imgAppPix, phi, 0, 0);

    // Convert pre-rotated pixmap to image and distort the image
    imgApparent = imgAppPix.toImage();
    distortThreaded();

    // Convert distorted image to pixmap, rotate and crop
    imgDistPix = QPixmap::fromImage(imgDistorted);
    imgDistPix = rotate(imgDistPix, -phi, 0, 0);
    QRect rect(wSize/2, wSize/2, wSize, wSize);
    imgDistPix = imgDistPix.copy(rect);

    imgActPix = QPixmap::fromImage(imgActual);

    // Scale pixmaps to fit in labels
    int labelH = ui->actLabel->height();
    imgDistPix = imgDistPix.scaled(labelH, labelH, Qt::KeepAspectRatio, Qt::SmoothTransformation);
    imgActPix = imgActPix.scaled(labelH, labelH, Qt::KeepAspectRatio, Qt::SmoothTransformation);

    double ratio = (double)labelH/wSize;
    // Draw grids and markers
    if (grid == true) {
        drawGrid(imgActPix);
//        drawGrid(imgAppPixDisp);
        drawGrid(imgDistPix);
        drawRadius(imgDistPix, ratio);
    }
    if (markers) {
        drawMarker(imgDistPix, ratio*(wSize/2 + apparentX), ratio*(wSize/2 - apparentY), 10, Qt::blue);
        drawMarker(imgDistPix, ratio*(wSize/2 + apparentX2), ratio*(wSize/2 - apparentY2), 10, Qt::blue);
        drawMarker(imgDistPix, ratio*(wSize/2 + actualX), ratio*(wSize/2 - actualY), 10, Qt::red);
    }

    // Draw legend after scaling to ensure text clarity
    if (legendCheck && markers) {
      drawLegend(imgDistPix, ui->distLabel->height());
    }

    // Draw pixmaps on QLabels
    ui->actLabel->setPixmap(imgActPix);
    ui->distLabel->setPixmap(imgDistPix);
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

void MainWindow::distort(int begin, int end) {
    int rows = imgDistorted.height();
    int cols = imgDistorted.width();
    // Evaluate each point in imgDistorted plane ~ lens plane
    for (int row = begin; row < end; row++) {
        for (int col = 0; col <= cols; col++) { // <= ???????????????????????????????????????


            double x = (col - apparentAbs - cols/2.0) * CHI;
            double y = (rows/2.0 - row) * CHI;

            // Calculate distance and angle of the point evaluated relative to center of lens (origin)
            double r = sqrt(x*x + y*y);
            double theta = atan2(y, x);

            std::pair<double, double> pos;

            if (mode == "infinite"){
                pos = pointMass(r, theta);
            }
            else if (mode == "finite"){
                pos = pointMassFinite(r, theta);
            }


            // Translate to array index
            int row_ = (int)round(rows/2.0 - pos.second);
            int col_ = (int)round(apparentAbs + cols/2.0 + pos.first);


            // If (x', y') within source, copy value to imgDistorted
            if (row_ < rows && col_ < cols && row_ > 0 && col_ >= 0) {
            imgDistorted.setPixel(col, row, imgApparent.pixel(col_, row_));
            }
        }
    }
}

std::pair<double, double> MainWindow::pointMass(double r, double theta){
    // Point mass lens equation
    double frac = (einsteinR * einsteinR * r) / (r * r + R * R + 2 * r * R * cos(theta));
    double x_ = (r*cos(theta) + frac * (r / R + cos(theta))) / CHI;
    double y_ = (r*sin(theta) - frac * sin(theta)) / CHI;
    return {x_, y_};
}

std::pair<double, double> MainWindow::pointMassFinite(double r, double theta){
    double xTemp = 0;
    double yTemp = 0;
    for(int m = 1; m <= terms; m++){
        int sign = 1 - 2*(m%2);
        double frac = std::pow((r/R), m);
        xTemp += sign * frac * std::cos(m*theta);
        yTemp -= sign * frac * std::sin(m*theta);
    }
    double x_ = (r * std::cos(theta) - einsteinR*einsteinR/R*xTemp)/CHI;
    double y_ = (r * std::sin(theta) - einsteinR*einsteinR/R*yTemp)/CHI;
    return {x_, y_};
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
            QPoint posApp(apparentX + imgApparent.width()/2 - rocket1.width()/2, -apparentY + imgApparent.height()/2 - rocket1.height()/2);
            QPoint posAct(actualX + imgActual.width()/2 - rocket1.width()/2, imgActual.height()/2 - actualY - rocket1.height()/2);
            pAct.drawPixmap(posAct, rocket1);
            pApp.drawPixmap(posApp, rocket1);
        }
        QRect rectApp(apparentX + imgApparent.width()/2 - srcSize, -apparentY + imgApparent.height()/2 - srcSize, 2*srcSize, 2*srcSize);
        QRect rectAct(actualX + imgActual.width()/2 - srcSize, imgActual.height()/2 - actualY - srcSize, 2*srcSize, 2*srcSize);
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

void MainWindow::drawRadius(QPixmap& src, double ratio){
    QPointF center(src.width()/2, src.height()/2);
    QPainter painter(&src);
    QPen pen(Qt::gray, 2, Qt::DashLine);
    painter.setPen(pen);
    painter.setOpacity(0.3);
    painter.drawEllipse(center, (int)round(einsteinR/CHI*ratio), (int)round(einsteinR/CHI*ratio));
}

void MainWindow::drawGrid(QPixmap& img){
    QPainter painter(&img);
    QPen pen(Qt::gray, 2, Qt::SolidLine);
    painter.setPen(pen);
    painter.setOpacity(0.3);

    if (gridSize > 0) {
        int remainder = (img.height()%gridSize)/2;
        for (int var = img.height()/gridSize; var < img.height()*1.5;) {
            QLineF lineVert(img.height()-var-remainder, 0, img.height()-var-remainder, img.height()-remainder);
            QLineF lineHor(0, img.height()-var-remainder, img.height()-remainder, img.height()-var-remainder);
            painter.drawLine(lineVert);
            painter.drawLine(lineHor);
            var+=img.height()/gridSize;
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

void MainWindow::drawLegend(QPixmap& img, int refSize){

    // Create legend pixmap
    int legendHeight = refSize/14;
    int legendWidth = refSize/4;
    QPixmap legend(legendWidth, legendHeight);

    // Background color of legend
    legend.fill(Qt::gray);

    // Draw markers in legend
    int markerSize = legendWidth/16;
    drawMarker(legend, legendWidth/10, legendHeight/2 - markerSize, markerSize, Qt::red);
    drawMarker(legend, legendWidth/10, legendHeight/2 + markerSize, markerSize, Qt::blue);

    int fontSize = legendWidth/11.5;
    drawText(legend, legendWidth/15 + markerSize*2, legendHeight/2 - markerSize + markerSize/2, fontSize, "Actual position");
    drawText(legend, legendWidth/15 + markerSize*2, legendHeight/2 + markerSize + markerSize/2, fontSize, "Apparent positions");

    // Set legend opacity and draw to main pixmap
    QPainter painter(&img);
    int xPos = 0;
    if (apparentX < -wSize/8 && apparentY > wSize/8){
        xPos += refSize - legendWidth;
    }
    painter.setOpacity(0.6);
    painter.drawPixmap(xPos, 0, legend);
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
        darkPalette.setColor(QPalette::ToolTipBase,QColor(42,42,42));
        qApp->setPalette(darkPalette);

    } else{
        qApp->setPalette(this->style()->standardPalette());
    }
}

void MainWindow::saveImage() {
    if (markers && legendCheck) {
        drawLegend(imgDistPix, wSize);
    }


    QString defaultFileName = QDate::currentDate().toString("'cosmoai_'yyyy-MM-dd");
    defaultFileName.append(QTime::currentTime().toString("-hh-mm-ss'.png'"));


    QImage imageDist = imgDistPix.toImage();
    QImage imageAct = imgActPix.toImage();

    QImage combined(2*imageDist.width(), imageDist.height(), QImage::Format_RGB32);
    QPainter painter(&combined);
    painter.drawImage(0, 0, imageAct);
    painter.drawImage(imageDist.width(), 0, imageDist);
    painter.end();

    QString fileName = QFileDialog::getSaveFileName(this, tr("Save Image"), QString(defaultFileName), tr("PNG (*.png)"));

    if (!fileName.isEmpty())
    {
        combined.save(fileName);
    }
}

void MainWindow::calculateStuff(){
    // Calculate positions and angles
    actualAbs = sqrt(actualX*actualX + actualY*actualY);
    double ratio1 = 0.5 + sqrt(0.25 + einsteinR*einsteinR/(CHI*CHI*actualAbs*actualAbs));
    double ratio2 = 0.5 - sqrt(0.25 + einsteinR*einsteinR/(CHI*CHI*actualAbs*actualAbs));
    apparentAbs = actualAbs*ratio1;
    apparentAbs2 = actualAbs*ratio2;
    apparentX = actualX * ratio1;
    apparentY = actualY * ratio1;
    apparentX2 = actualX * ratio2;
    apparentY2 = actualY * ratio2;
    R = apparentAbs * CHI;
    phi = atan2(actualY, actualX);

    // Spherical
    //    X = apparentX * CHI;
    //    Y = apparentY * CHI;
}

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

MainWindow::~MainWindow()
{
    delete ui;
}


void MainWindow::setup(){
    imgActual = QImage(wSize, wSize, QImage::Format_RGB32);
    imgApparent = QImage(2*wSize, 2*wSize, QImage::Format_RGB32);
    imgDistorted = QImage(2*wSize, 2*wSize, QImage::Format_RGB32);
    rocket = QPixmap(":/images/images/Tintin.png");
    lensType = "point";

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
    ui->termsSpinbox->setMaximum(100);

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
    CHI = 0.65;
    actualX = 0;
    actualY = 0;
    source = ui->srcTypeComboBox->currentText();

    // Set initial values for UI elements
    ui->einsteinSpinbox->setValue(einsteinR);
    ui->einsteinSlider->setSliderPosition(einsteinR);
    ui->srcSizeSpinbox->setValue(srcSize);
    ui->srcSizeSlider->setSliderPosition(srcSize);
    ui->lensDistSpinbox->setValue(CHI*100);
    ui->lensDistSlider->setSliderPosition(CHI*100);
    ui->xSpinbox->setValue(actualX);
    ui->xSlider->setSliderPosition(actualX);
    ui->ySpinbox->setValue(actualY);
    ui->ySlider->setSliderPosition(actualY);
    ui->gridBox->setChecked(grid);
    ui->markerBox->setChecked(markers);
    ui->actionMarkers->setChecked(markers);
    ui->actionLegend->setChecked(legendCheck);
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
    CHI = arg1/100.0;
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
    unsigned int input = QInputDialog::getInt(this,"Change resolution", "Current: " + currentSize, wSize, 50, 2048, 50);
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


void MainWindow::on_saveButton_clicked()
{
    saveImage();
}


void MainWindow::on_actionSave_image_as_triggered()
{
    saveImage();
}


void MainWindow::on_infTermsCheckbox_toggled(bool checked)
{
    if (checked) {
        mode = "infinite";
    } else {
        mode = "finite";
    }
    ui->termsSpinbox->setDisabled(checked);
    updateImg();
}


void MainWindow::on_termsSpinbox_valueChanged(int arg1)
{
    terms = arg1;
    updateImg();
}


//std::pair<double, double> Simulator::spherical(double r, double theta) const {
//    double ksi1 = 0;
//    double ksi2 = 0;

//    for (int m=1; m<n; m++){
//        double frac = pow(r, m) / factorial_(m);
//        double subTerm1 = 0;
//        double subTerm2 = 0;
//        for (int s=(m+1)%2; s<=m+1 && s<n; s+=2){
//            double alpha = alphas_val[m][s];
//            double beta = betas_val[m][s];
//            int c_p = 1 + s/(m + 1);
//            int c_m = 1 - s/(m + 1);
//            subTerm1 += 1.0/4*((alpha*cos((s-1)*theta) + beta*sin((s-1)*theta))*c_p + (alpha*cos((s+1)*theta) + beta*sin((s+1)*theta))*c_m);
//            subTerm2 += 1.0/4*((-alpha*sin((s-1)*theta) + beta*cos((s-1)*theta))*c_p + (alpha*sin((s+1)*theta) - beta*cos((s+1)*theta))*c_m);
//        }
//        double term1 = frac*subTerm1;
//        double term2 = frac*subTerm2;
//        ksi1 += term1;
//        ksi2 += term2;
//        // Break summation if term is less than 1/100 of ksi or if ksi is well outside frame
//        if ( ((std::abs(term1) < std::abs(ksi1)/100000) && (std::abs(term2) < std::abs(ksi2)/100000)) || (ksi1 < -100000*size || ksi1 > 100000*size || ksi2 < -100000*size || ksi2 > 100000*size) ){
//            break;
//        }
//    }
//    return {ksi1, ksi2};
//}

void MainWindow::on_actionAbout_triggered()
{
    QMessageBox::about(this, "About CosmoAI",
                       "<h2>"
                       "Test"
                       "</h2>"
                       "<br>"
                       "List:"
                       "<ul>"
                       "<li> Item 1 </li>"
                       "<li> Item 2 </li>"
                       "<li> Item 3 </li>"
                       "</ul>"
                       "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum."
                       );
}

