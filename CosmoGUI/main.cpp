#include "cosmogui.h"

#include <QApplication>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    CosmoGUI w;
    w.show();
    return a.exec();
}
