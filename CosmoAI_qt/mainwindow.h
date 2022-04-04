#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

private:
    bool grid;
    bool markers;
    int wSize;
    int einsteinR;
    int srcSize;
    int KL_percent;
    int xPos;
    int yPos;
    double phi;

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private:
    Ui::MainWindow *ui;
    QImage imgApparent;
    QImage imgActual;
    QImage imgDistorted;
    QPixmap rocket;
    QString source;

    void init_values();
    void drawGrid(QPixmap &img);
private slots:
//    void drawSource(QImage&, double, double);
//    void distort(QImage, QImage&, double, double, double);
    void updateImg();
//    void updateValues();

    void drawSourceThreaded(QImage&, double, double);
    void drawSource(int, int, QImage&, double, double);
    void distort(int, int, QImage, QImage&, double, double, double);
    void distortThreaded(double, double, QImage&, QImage&, double);

    void on_einsteinSpinbox_valueChanged();
    void on_srcSizeSpinbox_valueChanged();
    void on_lensDistSpinbox_valueChanged();
    void on_xSpinbox_valueChanged();
    void on_ySpinbox_valueChanged();
    void on_gridBox_stateChanged(int arg1);
    void on_markerBox_stateChanged(int arg1);
    void on_pushButton_clicked();
    void on_srcTypeComboBox_currentTextChanged(const QString &arg1);
};
#endif // MAINWINDOW_H
