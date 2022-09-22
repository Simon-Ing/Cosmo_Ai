/* (C) 2022: Hans Georg Schaathun <hg@schaathun.net> */

#include "Simulator.h"
#include <symengine/expression.h>
#include <symengine/lambda_double.h>
#include <thread>
#include <symengine/parser.h>
#include <fstream>

double factorial_(unsigned int n);

SphereSimulator::SphereSimulator() :
   Simulator::Simulator()
{ 
    std::cout << "Instantiating SphereSimulator ... \n" ;
    initAlphasBetas();
}
SphereSimulator::SphereSimulator(int s) :
   Simulator::Simulator(s)
{ 
    std::cout << "Instantiating SphereSimulator (size = " << s << ") ...\n" ;
    initAlphasBetas();
}

void SphereSimulator::initAlphasBetas() {

    auto x = SymEngine::symbol("x");
    auto y = SymEngine::symbol("y");
    auto g = SymEngine::symbol("g");
    auto c = SymEngine::symbol("c");

    std::string filename("50.txt");
    std::ifstream input;
    input.open(filename);

    if (!input.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }

    while (input) {
        std::string m, s;
        std::string alpha;
        std::string beta;
        std::getline(input, m, ':');
        std::getline(input, s, ':');
        std::getline(input, alpha, ':');
        std::getline(input, beta);
        if (input) {
            auto alpha_sym = SymEngine::parse(alpha);
            auto beta_sym = SymEngine::parse(beta);
            // The following two variables are unused.
            // SymEngine::LambdaRealDoubleVisitor alpha_num, beta_num;
            alphas_l[std::stoi(m)][std::stoi(s)].init({x, y, g, c}, *alpha_sym);
            betas_l[std::stoi(m)][std::stoi(s)].init({x, y, g, c}, *beta_sym);
        }
    }
}

void SphereSimulator::calculateAlphaBeta() {
    double GAMMA = einsteinR/2.0;

    // calculate all amplitudes for given X, Y, GAMMA, CHI
    // This is done here to before the code is parallellised
    for (int m = 1; m <= nterms; m++){
        for (int s = (m+1)%2; s <= (m+1); s+=2){
            alphas_val[m][s] = alphas_l[m][s].call({apparentAbs*CHI, 0, GAMMA, CHI});
            betas_val[m][s] = betas_l[m][s].call({apparentAbs*CHI, 0, GAMMA, CHI});
        }
    }
}

// Calculate the main formula for the SIS model
std::pair<double, double> SphereSimulator::getDistortedPos(double r, double theta) const {
    double ksi1 = r*cos(theta) ;
    double ksi2 = r*sin(theta) ;

    for (int m=1; m<=nterms; m++){
        double frac = pow(r, m) / factorial_(m);
        double subTerm1 = 0;
        double subTerm2 = 0;
        for (int s = (m+1)%2; s <= (m+1); s+=2){
            double alpha = alphas_val[m][s];
            double beta = betas_val[m][s];
            int c_p = 1 + s/(m + 1);
            int c_m = 1 - s/(m + 1);
            subTerm1 += 0.5*( (alpha*cos((s-1)*theta) + beta*sin((s-1)*theta))*c_p 
                            + (alpha*cos((s+1)*theta) + beta*sin((s+1)*theta))*c_m );
            subTerm2 += 0.5*( (-alpha*sin((s-1)*theta) + beta*cos((s-1)*theta))*c_p 
                            + (alpha*sin((s+1)*theta) - beta*cos((s+1)*theta))*c_m);
        }
        ksi1 += frac*subTerm1;
        ksi2 += frac*subTerm2;
    }
    return {ksi1, ksi2};
}

