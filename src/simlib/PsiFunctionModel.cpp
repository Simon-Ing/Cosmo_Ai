/* (C) 2023: Hans Georg Schaathun <georg@schaathun.net> */

#include "cosmosim/SampledLens.h"

#include <thread>
#include "simaux.h"


void PsiFunctionModel::updateApparentAbs( ) {
    std::cout << "[PsiFunctionModel] updateApparentAbs() updates psi.\n" ;
}
cv::Point2d PsiFunctionModel::calculateEta( cv::Point2d xi ) {
   cv::Point2d xy = cv::Point2d( -lens->psiYfunction( xi.x, xi.y ),
         -lens->psiXfunction( xi.x, xi.y ) ) ;
   return (xi - xy)/CHI ;
}
