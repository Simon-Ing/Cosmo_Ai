#include "cosmogui.h"
#include "./ui_cosmogui.h"

bool grid = false;
bool markers = false;
double srcSize = 0.0;
double lensDist = 0.0;
double lensPos = 0.0;


CosmoGUI::CosmoGUI(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::CosmoGUI)
{
    ui->setupUi(this);

    connect(ui->srcSizeSpinbox, SIGNAL(valueChanged(int)), ui->srcSizeSlider, SLOT(setValue(int)));
    connect(ui->srcSizeSlider, SIGNAL(valueChanged(int)), ui->srcSizeSpinbox, SLOT(setValue(int)));
    connect(ui->lensDistSpinbox, SIGNAL(valueChanged(int)), ui->lensDistSlider, SLOT(setValue(int)));
    connect(ui->lensDistSlider, SIGNAL(valueChanged(int)), ui->lensDistSpinbox, SLOT(setValue(int)));
    connect(ui->lensPosSpinbox, SIGNAL(valueChanged(int)), ui->lensPosSlider, SLOT(setValue(int)));
    connect(ui->lensPosSlider, SIGNAL(valueChanged(int)), ui->lensPosSpinbox, SLOT(setValue(int)));


}

CosmoGUI::~CosmoGUI()
{
    delete ui;
}
