#include "cosmogui.h"
#include "./ui_cosmogui.h"

CosmoGUI::CosmoGUI(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::CosmoGUI)
{
    ui->setupUi(this);


}

CosmoGUI::~CosmoGUI()
{
    delete ui;
}

