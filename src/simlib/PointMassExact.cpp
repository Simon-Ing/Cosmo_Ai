/* (C) 2023: Hans Georg Schaathun <georg@schaathun.net> */

#include "cosmosim/Simulator.h"

/* The following is a default implementation for the point mass lens. 
 * It would be better to make the class abstract and move this definition to the 
 * subclass. */
cv::Point2d PointMassExact::getDistortedPos(double r, double theta) const {
    double R = getXiAbs() ;
    double einsteinR = lens->getEinsteinR() ;
    double frac = (einsteinR * einsteinR * r) / (r * r + R * R + 2 * r * R * cos(theta));
    double nu1 = r*cos(theta) + frac * (r / R + cos(theta)) ;
    double nu2 = r*sin(theta) - frac * sin(theta) ;
    return cv::Point2d( nu1/CHI, nu2/CHI ) ;
}

