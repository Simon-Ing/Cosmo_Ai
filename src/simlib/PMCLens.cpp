/* (C) 2022: Hans Georg Schaathun <georg@schaathun.net> */

#include "cosmosim/PixMap.h"

#include <thread>
#include <fstream>

/* The following is a default implementation for the point mass lens. 
 * It would be better to make the class abstract and move this definition to the 
 * subclass. */
cv::Point2f PointMassLens::getDistortedPos(double r, double theta) const {
    double R = apparentAbs * CHI ;
    double frac = (einsteinR * einsteinR * r) / (r * r + R * R + 2 * r * R * cos(theta));
    double nu1 = r*cos(theta) + frac * (r / R + cos(theta)) ;
    double nu2 = r*sin(theta) - frac * sin(theta) ;
    return cv::Point2f( nu1/CHI, nu2/CHI ) ;
}

void PointMassLens::updateApparentAbs( ) {
    // The apparent position is the solution to a quadratic equation.
    // thus there are two solutions.
    double root = sqrt(0.25*getEtaSquare() + einsteinR*einsteinR/(CHI*CHI));

    tentativeCentre = apparentAbs = getEtaAbs()/2 + root ;
    apparentAbs2 = getEtaAbs()/2 - root ;
}
