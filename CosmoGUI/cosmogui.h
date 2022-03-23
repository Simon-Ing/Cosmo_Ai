#ifndef COSMOGUI_H
#define COSMOGUI_H

#include <QMainWindow>
#include <opencv2/opencv.hpp>

QT_BEGIN_NAMESPACE
namespace Ui { class CosmoGUI; }
QT_END_NAMESPACE

class CosmoGUI : public QMainWindow
{
    Q_OBJECT

public:
    CosmoGUI(QWidget *parent = nullptr);
    ~CosmoGUI();

    QImage imdisplay;  //This will create QImage which is shown in Qt label
    QTimer* Timer;   // A timer is needed in GUI application


private:
    Ui::CosmoGUI *ui;


public slots:
    void refLines(cv::Mat&);
    void drawSource(cv::Mat&, int, int);
    void distort(int, int, int, int, cv::Mat, cv::Mat&, double);
    void updateImg();
    void updateValues();

private slots:

};
#endif // COSMOGUI_H
