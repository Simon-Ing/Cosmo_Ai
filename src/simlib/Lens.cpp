/* (C) 2024: Hans Georg Schaathun <georg@schaathun.net> */

#include "cosmosim/Lens.h"
#include "simaux.h"

cv::Point2d Lens::getXi( cv::Point2d eta ) {

   cv::Point2d xi0, xi1 = eta ;
   int cont = 1, count = 0, maxcount = 200 ;
   double dist, dist0=pow(10,12), threshold = 0.02 ;

   if (DEBUG) std::cout << "[Lens::getXi] " << eta << "\n" ;

   /** This block makes a fix-point iteration to find \xi. */
   while ( cont ) {
      xi0 = xi1 ;
      double x = psiXvalue( xi0.x, xi0.y ),
             y = psiYvalue( xi0.x, xi0.y ) ;
      if (DEBUG) std::cout
	   << "[Lens] Fix pt it'n " << count
           << "; xi0=" << xi0 << "; Delta eta = " << x << ", " << y << "\n" ;
      if ( std::isnan( x ) ) x = 0 ;
      if ( std::isnan( y ) ) y = 0 ;
      if ( std::isinf( x ) || std::isinf( y ) ) {
         cont = 0 ; break ;
      }
      xi1 = eta + cv::Point2d( x, y ) ;
      dist = cv::norm( cv::Mat(xi1-xi0), cv::NORM_L2 ) ;
      if ( dist < threshold ) cont = 0 ;
      if ( ++count > maxcount ) cont = 0 ;
   }
   if (DEBUG) {
      if ( dist > threshold ) {
         std::cout << "Bad approximation of xi: xi0=" << xi0 
            << "; xi1=" << xi1 << "; dist=" << dist << "\n" ;
      } else {
         std::cout << "[Lens] Good approximation: xi0=" << xi0 
            << "; xi1=" << xi1 << "\n" ;
      }
   }
   if (DEBUG) std::cout << "[Lens::getXi] " << eta << " -> " << xi1 << "\n" ;
   return xi1 ;
}

void Lens::calculateAlphaBeta( cv::Point2d, int ) { throw NotImplemented() ; }
double Lens::getAlphaXi( int m, int s ) { throw NotImplemented() ; }
double Lens::getBetaXi( int m, int s ) { throw NotImplemented() ; }
double Lens::getAlpha( cv::Point2d xi, int m, int s ) { throw NotImplemented() ; }
double Lens::getBeta( cv::Point2d xi, int m, int s ) { throw NotImplemented() ; }

double Lens::criticalXi( double phi ) const {
   cv::Point2d xi ; 
   int cont = 1, count = 0, maxcount = 200 ;
   double dist, dist0=pow(10,12), threshold = 0.02 ;

   double r0, r1 = 10 ;

   /** This block makes a fix-point iteration to find \xi. */
   while ( cont ) {
      r0 = r1 ;
      xi = r0*cv::Point2d( cos( phi ), sin( phi ) ) ;
      double x = psiXvalue( xi.x, xi.y ),
             y = psiYvalue( xi.x, xi.y ) ;
      if (DEBUG) std::cout
	   << "[Lens] Fix pt it'n " << count << "; r0=" << r0 << std::endl ;
      r1 = sqrt( x * x + y * y ) ;
      dist = r0 - r1 ;
      if ( dist < threshold ) cont = 0 ;
      if ( ++count > maxcount ) cont = 0 ;
   }
   if (DEBUG) {
      if ( dist > threshold ) {
         std::cout << "[Lens::criticalCurve] Bad approximation: r0=" << r0 
            << "; r1=" << r1 << "; dist=" << dist << "\n" ;
      } else {
         std::cout << "[Lens::criticalCurve] [Lens] Good approximation: r0=" << r0 
            << "; r1=" << r1 << "\n" ;
      }
   }
   return r1 ;
}
cv::Point2d Lens::caustic( double phi ) const { throw NotImplemented() ; }
double Lens::psiValue( double x, double y ) const { throw NotImplemented() ; }
double Lens::psiXvalue( double x, double y ) const { throw NotImplemented() ; }
double Lens::psiYvalue( double x, double y ) const { throw NotImplemented() ; }

std::string Lens::idString() {
   return "Lens (Superclass)" ;
};

double Lens::getAlphaPy( double x, double y, int m, int s ) {
  return  Lens::getAlpha( cv::Point2d(x,y), m, s ) ;
}
double Lens::getBetaPy( double x, double y, int m, int s ) {
  return  Lens::getBeta( cv::Point2d(x,y), m, s ) ;
}

