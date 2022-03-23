#include "cosmogui.h"
#include "iostream"

#include <QApplication>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    CosmoGUI w;
    w.show();
    std::cout << "hello world" << std::endl;

    return a.exec();
}
