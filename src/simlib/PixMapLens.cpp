/* (C) 2022: Hans Georg Schaathun <georg@schaathun.net> */

#include "cosmosim/PixMap.h"

#include <symengine/expression.h>
#include <symengine/lambda_double.h>
#include <symengine/parser.h>

#include <thread>
#include <fstream>

cv::Mat PixMapLens::getPsi() {
   return psi ;
}
cv::Mat PixMapLens::getMassMap() {
   return massMap ;
}
cv::Mat PixMapLens::getEinsteinMap() {
   return einsteinMap ;
}
void PixMapLens::setEinsteinMap( cv::Mat map ) {
   einsteinMap = map ;
   // Calculate psi and massMap here
}

