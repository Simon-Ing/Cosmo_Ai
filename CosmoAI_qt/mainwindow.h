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
    Ui::MainWindow *ui;
    bool grid;
    bool markers;
    bool legendCheck;
    bool darkMode = true;
    int gridSize;
    int wSize = 600;
    int wSizeWide;
    int einsteinR;
    int srcSize;
    int actualX;
    int actualY;
    double phi;
    double CHI;
    double actualAbs;
    double apparentAbs;
    double apparentX;
    double apparentY;
    double apparentAbs2;
    double R;
    int apparentX2;
    int apparentY2;
    QImage imgApparent;
    QImage imgActual;
    QImage imgDistorted;
    QPixmap pixApp;
    QPixmap pixAct;
    QPixmap pixDist;
    QPixmap imgAppPix;
    QPixmap imgActPix;
    QPixmap imgDistPix;
    QPixmap rocket;
    QPixmap legend;
    QString source;
    QString lensType;
    int terms = 1;
    std::string mode = "finite";

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private:
    void init_values();
    void drawGrid(QPixmap &img);
    void drawRadius(QPixmap& src, double);
    void drawMarker(QPixmap &src, int x, int y, int size, QColor color);
    void setup();
    void updateImg();
    void drawGaussianThreaded(QImage&, double, double);
    void drawGaussian(int, int, QImage&, double, double);
    void distort(int, int);
    void distortThreaded();
    QPixmap rotate(QPixmap src, double angle, int x, int y);
    void resizeEvent(QResizeEvent *event);
    void drawLegend(QPixmap&, int refSize);
    void drawText(QPixmap& img, int x, int y, int fontSize, QString text);
    void theme();
    void drawSource();
    void saveImage();
    std::pair<double, double> pointMass(double r, double theta);
    std::pair<double, double> spherical(double r, double theta) const;
    std::pair<double, double> pointMassFinite(double r, double theta);

    void calculateStuff();
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
    void on_actionReset_triggered();
    void on_actionMarkers_toggled(bool arg1);
    void on_actionLegend_toggled(bool arg1);
    void on_actionOff_triggered();
    void on_action2x2_triggered();
    void on_action4x4_triggered();
    void on_action8x8_triggered();
    void on_action12x12_triggered();
    void on_actionChange_resolution_triggered();
    void on_actionCustom_triggered();
    void on_actionDark_mode_toggled(bool arg1);
    void on_saveButton_clicked();
    void on_actionSave_image_as_triggered();
    void on_infTermsCheckbox_toggled(bool checked);
    void on_termsSpinbox_valueChanged(int arg1);
    void on_actionAbout_triggered();
};

#endif // MAINWINDOW_H
