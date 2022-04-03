#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private:
    Ui::MainWindow *ui;
    QImage imgApparent;
    QImage imgActual;
    QImage imgDistorted;
    QTimer* Timer;



private slots:
//    void refLines(QImage&);
    void drawSource(int, int, QImage&, double, double);
//    void drawParallel(cv::Mat&, double, double);
    void distort(int, int, double, double, QImage, QImage&, double);
//    void parallel(double, double, cv::Mat&, cv::Mat&, double);
    void updateImg();
    void updateValues();

};
#endif // MAINWINDOW_H
