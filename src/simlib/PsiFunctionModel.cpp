/* (C) 2023: Hans Georg Schaathun <georg@schaathun.net> */

#include "cosmosim/SampledLens.h"

#include <thread>
#include "simaux.h"


void PsiFunctionModel::setPsiFunctionLens( PsiFunctionLens *l ) {
   lens = psilens = l ;
}
cv::Point2d PsiFunctionModel::calculateEta( cv::Point2d xi ) {
   cv::Point2d xy = cv::Point2d( lens->psiXvalue( xi.x, xi.y ),
         lens->psiYvalue( xi.x, xi.y ) ) ;
   return (xi - xy)/CHI ;
}
