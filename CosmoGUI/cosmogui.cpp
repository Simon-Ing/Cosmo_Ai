#include "cosmogui.h"
#include "./ui_cosmogui.h"
#include <opencv2/opencv.hpp>
#include <thread>
#include <random>
#include <string>
#define _USE_MATH_DEFINES // for C++
#include <math.h>
#include<QDebug>


bool grid = false;
bool markers = false;
int wSize = 600;
int einsteinR = 0;
int srcSize = 0;
int lensDist = 0;
int xPos = 0;
int yPos = 0;


CosmoGUI::CosmoGUI(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::CosmoGUI)
{
    ui->setupUi(this);

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

    connect(ui->einsteinSpinbox, SIGNAL(valueChanged(int)), ui->einsteinSlider, SLOT(setValue(int)));
    connect(ui->einsteinSlider, SIGNAL(valueChanged(int)), ui->einsteinSpinbox, SLOT(setValue(int)));
    connect(ui->srcSizeSpinbox, SIGNAL(valueChanged(int)), ui->srcSizeSlider, SLOT(setValue(int)));
    connect(ui->srcSizeSlider, SIGNAL(valueChanged(int)), ui->srcSizeSpinbox, SLOT(setValue(int)));
    connect(ui->lensDistSpinbox, SIGNAL(valueChanged(int)), ui->lensDistSlider, SLOT(setValue(int)));
    connect(ui->lensDistSlider, SIGNAL(valueChanged(int)), ui->lensDistSpinbox, SLOT(setValue(int)));
    connect(ui->xSpinbox, SIGNAL(valueChanged(int)), ui->xSlider, SLOT(setValue(int)));
    connect(ui->xSlider, SIGNAL(valueChanged(int)), ui->xSpinbox, SLOT(setValue(int)));
    connect(ui->ySpinbox, SIGNAL(valueChanged(int)), ui->ySlider, SLOT(setValue(int)));
    connect(ui->ySlider, SIGNAL(valueChanged(int)), ui->ySpinbox, SLOT(setValue(int)));

}


CosmoGUI::~CosmoGUI()
{
    getVariableValues();
    delete ui;

}

void CosmoGUI::getVariableValues(){
    einsteinR = ui->einsteinSpinbox->value();
    srcSize = ui->srcSizeSpinbox->value();
    lensDist = ui->lensDistSpinbox->value();
    xPos = ui->xSpinbox->value();
    yPos = ui->ySpinbox->value();
    qDebug() << einsteinR << srcSize << lensDist << xPos << yPos;
}

