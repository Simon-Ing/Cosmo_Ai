#ifndef COSMOGUI_H
#define COSMOGUI_H

#include <QMainWindow>

QT_BEGIN_NAMESPACE
namespace Ui { class CosmoGUI; }
QT_END_NAMESPACE

class CosmoGUI : public QMainWindow
{
    Q_OBJECT

public:
    CosmoGUI(QWidget *parent = nullptr);
    ~CosmoGUI();

private:
    Ui::CosmoGUI *ui;
    void getVariableValues();
};
#endif // COSMOGUI_H
