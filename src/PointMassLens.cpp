/* (C) 2022: Hans Georg Schaathun <hg@schaathun.net> */

#include "Simulator.h"
#include <symengine/expression.h>
#include <symengine/lambda_double.h>
#include <thread>
#include <symengine/parser.h>
#include <fstream>

/* The following is a default implementation for the point mass lens. 
 * It would be better to make the class abstract and move this definition to the 
 * subclass. */
std::pair<double, double> PointMassLens::getDistortedPos(double r, double theta) const {
    double R = apparentAbs * CHI ;
    double frac = (einsteinR * einsteinR * r) / (r * r + R * R + 2 * r * R * cos(theta));
    double nu1 = r*cos(theta) + frac * (r / R + cos(theta)) ;
    double nu2 = r*sin(theta) - frac * sin(theta) ;
    nu1 /= CHI ;
    nu2 /= CHI ;
    return {nu1, nu2};
}
