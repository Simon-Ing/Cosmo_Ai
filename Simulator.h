//
// Created by simon on 07.04.2022.
//

#ifndef SPHERICAL_SIMULATOR_H
#define SPHERICAL_SIMULATOR_H

#include "opencv2/opencv.hpp"
#include <symengine/expression.h>
#include <symengine/lambda_double.h>

using namespace SymEngine;


class Simulator {
private:
    int mode;
    double GAMMA;
    double CHI;
    double actualX{};
    double actualY{};
    double apparentX{};
    double apparentY{};
    double apparentX2{};
    double apparentY2{};
    double X{};
    double Y{};
    double phi{};
    double actualAbs{};
    double apparentAbs{};
    double apparentAbs2{};
    double R{};
    int n;

    cv::Mat imgDistorted;

    std::array<std::array<LambdaRealDoubleVisitor, 52>, 51> alphas_l;
    std::array<std::array<LambdaRealDoubleVisitor, 52>, 51> betas_l;
    std::array<std::array<double, 52>, 51> alphas_val;
    std::array<std::array<double, 52>, 51> betas_val;

public:
    int size;
    std::string name;
    int CHI_percent;
    int sourceSize;
    int einsteinR;
    int xPosSlider;
    int yPosSlider;


public:
    Simulator();

    void update();

    void initGui();

    void writeToPngFiles(int);

private:
    void calculate();

//    void drawSource(cv::Mat& img, double xPos, double yPos) const;

    [[nodiscard]] std::pair<double, double> pointMass(double r, double theta) const;

    static void update_dummy(int, void*);

    cv::Mat formatImg(cv::Mat &imgDistorted, cv::Mat &imgActual, int displaySize) const;

    static void refLines(cv::Mat &target);

    void distort(int row, int col, const cv::Mat &src, cv::Mat &dst);

    void parallelDistort(const cv::Mat &src, cv::Mat &dst);

    void initAlphasBetas();

    std::pair<double, double> spherical(double r, double theta) const;


    void drawParallel(cv::Mat &img, int xPos, int yPos);

    void drawSource(int begin, int end, cv::Mat &img, int xPos, int yPos);
};


#endif //SPHERICAL_SIMULATOR_H