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
    int wSizeWide;
    int einsteinR;
    int srcSize;
    int xPos;
    int yPos;
    double phi;
    double KL;
    double actualPos;
    double apparentPos;
    double apparentPos2;
    double R;
    int actualX;
    int actualY;
    int apparentX;
    int apparentY;
    int apparentX2;
    int apparentY2;    Ui::MainWindow *ui;
    QImage imgApparent;
    QImage imgActual;
    QImage imgDistorted;
    QPixmap pixApp;
    QPixmap pixAct;
    QPixmap pixDist;
    QPixmap rocket;
    QString source;

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private:
    void init_values();
    void drawGrid(QPixmap &img);
    void drawRadius(QPixmap& src);
    void drawMarker(QPixmap &src, int x, int y, QColor color);
    void setup();
    void updateImg();
    void drawSourceThreaded(QImage&, double, double);
    void drawSource(int, int, QImage&, double, double);
    void distort(int, int);
    void distortThreaded();
    QPixmap rotate(QPixmap src, double angle, int x, int y);

private slots:
    void on_einsteinSpinbox_valueChanged();
    void on_srcSizeSpinbox_valueChanged();
    void on_lensDistSpinbox_valueChanged(int);
    void on_xSpinbox_valueChanged();
    void on_ySpinbox_valueChanged();
    void on_gridBox_stateChanged(int arg1);
    void on_markerBox_stateChanged(int arg1);
    void on_resetButton_clicked();
    void on_srcTypeComboBox_currentTextChanged(const QString &arg1);
};

#endif // MAINWINDOW_H
