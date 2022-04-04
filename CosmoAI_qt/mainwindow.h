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

private slots:
//    void refLines(QImage&);
    void drawSource(QImage&, double, double);
//    void drawParallel(cv::Mat&, double, double);
    void distort(QImage, QImage&, double, double, double);
//    void parallel(double, double, cv::Mat&, cv::Mat&, double);
    void updateImg();
    void updateValues();

    void on_einsteinSpinbox_valueChanged();
    void on_srcSizeSpinbox_valueChanged();
    void on_lensDistSpinbox_valueChanged();
    void on_xSpinbox_valueChanged();
    void on_ySpinbox_valueChanged();
};
#endif // MAINWINDOW_H
