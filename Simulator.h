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
    int size;
    int sourceSize;
    int mode;
    int einsteinR;
    double GAMMA;
    int xPosSlider;
    int yPosSlider;
    int CHI_percent;
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
    const static int n = 8;
    RCP<const Symbol> xSym, ySym, gammaSym, chiSym;
    std::vector<std::vector<RCP<const Basic>>> alphas;
    std::vector<std::vector<RCP<const Basic>>> betas;

    std::array<std::array<LambdaRealDoubleVisitor, n>, n> alphas_l;
    std::array<std::array<LambdaRealDoubleVisitor, n>, n> betas_l;

public:
    Simulator();

    void update();

    void initGui();

private:
    void calculate();

    void drawSource(cv::Mat& img, double xPos, double yPos) const;

    void parallelDistort(const cv::Mat& src, cv::Mat& dst);

    void distort(unsigned int begin, unsigned int end, const cv::Mat& src, cv::Mat& dst);

    [[nodiscard]] std::pair<double, double> pointMass(double r, double theta) const;

    std::pair<double, double> spherical(double r, double theta, std::array<std::array<LambdaRealDoubleVisitor, n>, n>&, std::array<std::array<LambdaRealDoubleVisitor, n>, n>&) const;

    void initAlphasBetas(std::array<std::array<LambdaRealDoubleVisitor, n>, n>& alphas_lambda, std::array<std::array<LambdaRealDoubleVisitor, n>, n>& betas_lambda);

    static void update_dummy(int, void*);

};


#endif //SPHERICAL_SIMULATOR_H
