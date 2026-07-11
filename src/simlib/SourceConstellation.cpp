/* (C) 2024: Hans Georg Schaathun <georg@schaathun.net> */

#include "cosmosim/Source.h"

SourceConstellation::~SourceConstellation() { 
   std::cout << "SourceConstellation::destructor]] does nothing\n" ;
   /*
   for ( int i=0 ; i<nsrc ; ++i ) {
      std::cout << "SourceConstellation::destructor]] " << i << "\n" ;
      delete src[i] ;
   }
   std::cout << "SourceConstellation::destructor]] terminates\n" ;
   */
}

void SourceConstellation::addSource( Source *l, double x, double y ) {
   std::cout << "SourceConstellation::addSource]] " << cv::Point2d(x,y) << "\n" ;
   this->xshift[this->nsrc] = x ;
   this->yshift[this->nsrc] = y ;
   this->src[this->nsrc++] = l ;
   std::cout << "SourceConstellation::addSource]] returns\n" ;
}
void SourceConstellation::drawParallel(cv::Mat& dst){
   std::cout << "SourceConstellation::drawParallel]]\n" ;
    for ( int i=0 ; i<nsrc ; ++i ) {
       std::cout << "SourceConstellation::drawParallel]] Constituent source no. " 
          << i << "\n" ;
       cv::Mat tr = (cv::Mat_<double>(2,3) << 1, 0, xshift[i], 0, 1, yshift[i]);
       std::cout << "SourceConstellation::drawParallel]] " << src << "\n" ;
       cv::Mat s = src[i]->getImage() ;
       std::cout << "SourceConstellation::drawParallel]] getImage() from constituent source has returned.\n" ;
       cv::Mat tmp = cv::Mat::zeros(s.size(), s.type());
       cv::warpAffine(s, tmp, tr, s.size()) ;
       cv::add( dst, tmp, dst ) ;
    }
   std::cout << "SourceConstellation::drawParallel]] returns\n" ;
}
