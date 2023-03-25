/* (C) 2023: Hans Georg Schaathun <georg@schaathun.net> *
 * Building on code by Simon Ingebrigtsen, Sondre Westbø Remøy,
 * Einar Leite Austnes, and Simon Nedreberg Runde
 */

#include "cosmosim/Simulator.h"
#include "simaux.h"

#include <thread>

#define DEBUG 0

double factorial_(unsigned int n);

LensModel::LensModel() :
        LensModel(false)
{ }
LensModel::LensModel(bool centred) :
        CHI(0.5),
        einsteinR(20),
        nterms(10),
        centredMode(centred),
        source(NULL)
{ }
LensModel::~LensModel() {
   std::cout << "Destruct lens model\n" ;
   delete source ;
}

/* Getters for the images */
cv::Mat LensModel::getActual() const {
   cv::Mat imgApparent = getSource() ;
   cv::Mat imgActual 
        = cv::Mat::zeros(imgApparent.size(), imgApparent.type());
   cv::Mat tr = (cv::Mat_<double>(2,3) << 1, 0, getEta().x, 0, 1, -getEta().y);
   std::cout << "getActual() (x,y)=(" << getEta().x << "," << getEta().y << ")\n" ;
   cv::warpAffine(imgApparent, imgActual, tr, imgApparent.size()) ;
   return imgActual ; 

}
cv::Mat LensModel::getSource() const {
   return source->getImage() ;
}
cv::Mat LensModel::getApparent() const {
   cv::Mat src, dst ;
   src = source->getImage() ;
   if ( rotatedMode ) {
       int nrows = src.rows ;
       int ncols = src.cols ;
       cv::Mat rot = cv::getRotationMatrix2D(cv::Point(nrows/2, ncols/2),
             360-phi*180/PI, 1) ;
       cv::warpAffine(src, dst, rot, src.size() ) ;
      return dst ;
   } else {
      return src ;
   }
}
cv::Mat LensModel::getDistorted() const {
   return imgDistorted ;
}

void LensModel::updateSecondary( ) {
   if ( apparentAbs2 == 0 ) {
      throw NotSupported() ;
   }
   throw NotImplemented() ;
   return updateInner() ;
}
void LensModel::update( ) {
   updateApparentAbs() ;
   return updateInner() ;
}
void LensModel::update( cv::Point2d xi ) {
   setXi( xi ) ;
   return updateInner() ;
}
void LensModel::updateInner( ) {
    cv::Mat imgApparent = getApparent() ;

    std::cout << "[LensModel::update()] x=" << eta.x << "; y= " << eta.y 
              << "; R=" << getEtaAbs() << "; theta=" << phi
              << "; R_E=" << einsteinR << "; CHI=" << CHI << "\n" ;
    std::cout << "[LensModel::update()] xi=" << getXi()   << "\n" ;
    std::cout << "[LensModel::update()] eta=" << getEta() << "; etaOffset=" << etaOffset << "\n" ;
    std::cout << "[LensModel::update()] nu=" << getNu()   << "\n" ;
    std::cout << "[LensModel::update()] centre=" << getCentre() << "\n" ;

    auto startTime = std::chrono::system_clock::now();

    this->calculateAlphaBeta() ;

    if ( rotatedMode ) {
       int nrows = imgApparent.rows ;
       int ncols = imgApparent.cols ;

       // Make Distorted Image
       // We work in a double sized image to avoid cropping
       cv::Mat imgD = cv::Mat(nrows*2, ncols*2, imgApparent.type(), 0.0 ) ;
       parallelDistort(imgApparent, imgD);
   
       // Correct the rotation applied to the source image
       cv::Mat rot = cv::getRotationMatrix2D(cv::Point(nrows, ncols), phi*180/PI, 1);
       std::cout << "rotatedMode=true\n" << rot << "\n" ;
       cv::warpAffine(imgD, imgD, rot, cv::Size(2*nrows, 2*ncols));    
           // crop distorted image
       imgDistorted = imgD(cv::Rect(nrows/2, ncols/2, nrows, ncols)) ;
    } else {
       imgDistorted = cv::Mat::zeros(imgApparent.size(), imgApparent.type()) ;
       parallelDistort(imgApparent, imgDistorted);
    }

    std::cout << "update() (x,y) = (" << eta.x << ", " << eta.y << ")\n" ;

    // Calculate run time for this function and print diagnostic output
    auto endTime = std::chrono::system_clock::now();
    std::cout << "Time to update(): " 
              << std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count() 
              << " milliseconds" << std::endl;

}


/* This just splits the image space in chunks and runs distort() in parallel */
void LensModel::parallelDistort(const cv::Mat& src, cv::Mat& dst) {
    int n_threads = std::thread::hardware_concurrency();
    if ( DEBUG ) std::cout << "Running with " << n_threads << " threads.\n" ;
    std::vector<std::thread> threads_vec;
    double maskRadius = getMaskRadius() ;
    int lower=0, rng=dst.rows, rng1 ; 
    if ( maskMode ) {
        double mrng ;
        cv::Point2d origin = getCentre() ;
        cv::Point2d ij = imageCoordinate( origin, dst ) ;
        std::cout << "mask " << ij << " - " << origin << "\n" ;
        lower = floor( ij.x - maskRadius ) ;
        if ( lower < 0 ) lower = 0 ;
        mrng = dst.rows - lower ;
        rng = ceil( 2.0*maskRadius ) + 1 ;
        if ( rng > mrng ) rng = mrng ;
        std::cout << maskRadius << " - " << lower << "/" << rng << "\n" ;
    } else {
        std::cout << "[LensModel] No mask \n" ;
    } 
    rng1 = ceil( rng/ n_threads ) ;
    for (int i = 0; i < n_threads; i++) {
        int begin = lower+rng1*i, end = begin+rng1 ;
        std::thread t([begin, end, src, &dst, this]() { distort(begin, end, src, dst); });
        threads_vec.push_back(std::move(t));
    }

    for (auto& thread : threads_vec) {
        thread.join();
    }
}


void LensModel::distort(int begin, int end, const cv::Mat& src, cv::Mat& dst) {
    // Iterate over the pixels in the image distorted image.
    // (row,col) are pixel co-ordinates
    double maskRadius = getMaskRadius()*CHI ;
    cv::Point2d xi = getXi() ;
    for (int row = begin; row < end; row++) {
        for (int col = 0; col < dst.cols; col++) {

            cv::Point2d pos, ij ;

            // Set coordinate system with origin at the centre of mass
            // in the distorted image in the lens plane.
            double x = (col - dst.cols / 2.0 ) * CHI - xi.x;
            double y = (dst.rows / 2.0 - row ) * CHI - xi.y;
            // (x,y) are coordinates in the lens plane, and hence the
            // multiplication by CHI

            // Calculate distance and angle of the point evaluated 
            // relative to CoM (origin)
            double r = sqrt(x * x + y * y);

            if ( maskMode && r > maskRadius ) {
            } else {
              double theta = x == 0 ? PI/2 : atan2(y, x);
              pos = this->getDistortedPos(r, theta);
              pos += etaOffset ;

              // Translate to array index in the source plane
              ij = imageCoordinate( pos, src ) ;
  
              // If the pixel is within range, copy value from src to dst
              if (ij.x < src.rows && ij.y < src.cols && ij.x >= 0 && ij.y >= 0) {
                 if ( 3 == src.channels() ) {
                    dst.at<cv::Vec3b>(row, col) = src.at<cv::Vec3b>( ij );
                 } else {
                    dst.at<uchar>(row, col) = src.at<uchar>( ij );
                 }
              }
            }
        }
    }
}

/* Initialiser.  The default implementation does nothing.
 * This is correct for any subclass that does not need the alpha/beta tables. */
void LensModel::calculateAlphaBeta() { }


/** *** Setters *** */

/* A.  Mode setters */
void LensModel::setMaskMode(bool b) {
   maskMode = b ; 
}
void LensModel::setBGColour(int b) { bgcolour = b ; }
void LensModel::setCentred(bool b) { centredMode = b ; }

/* B. Source model setter */
void LensModel::setSource(Source *src) {
    if ( source != NULL ) delete source ;
    source = src ;
}

/* C. Lens Model setter */
void LensModel::setNterms(int n) {
   nterms = n ;
}
void LensModel::setCHI(double chi) {
   CHI = chi ;
}
void LensModel::setEinsteinR(double r) {
   einsteinR = r ;
}

/* D. Position (eta) setters */

/* Re-calculate co-ordinates using updated parameter settings from the GUI.
 * This is called from the update() method.                                  */
void LensModel::setXY( double X, double Y, double chi, double er ) {

    CHI = chi ;
    einsteinR = er ;
    // Actual position in source plane
    eta = cv::Point2d( X, Y ) ;

    // Calculate Polar Co-ordinates
    phi = atan2(eta.y, eta.x); // Angle relative to x-axis

    std::cout << "[setXY] eta.y=" << eta.y 
              << "; actualY=" << Y 
              << "; eta=" << eta 
              << "\n" ;

    std::cout << "[setXY] Set position x=" << eta.x << "; y=" << eta.y
              << "; R=" << getEtaAbs() << "; theta=" << phi << ".\n" ;
}

/* Re-calculate co-ordinates using updated parameter settings from the GUI.
 * This is called from the update() method.                                  */
void LensModel::setPolar( double R, double theta, double chi, double er ) {

    CHI = chi ;
    einsteinR = er ;

    phi = PI*theta/180 ;

    // Actual position in source plane
    eta = cv::Point2d( R*cos(phi), R*sin(phi) ) ;

    std::cout << "[setPolar] Set position x=" << eta.x << "; y=" << eta.y
              << "; R=" << getEtaAbs() << "; theta=" << phi << ".\n" ;

}


/* Masking */
void LensModel::maskImage( ) {
    maskImage( imgDistorted, 1 ) ;
}
void LensModel::maskImage( double scale ) {
    maskImage( imgDistorted, scale ) ;
}
void LensModel::markMask( ) {
    markMask( imgDistorted ) ;
}
void LensModel::maskImage( cv::InputOutputArray r, double scale ) {
   throw NotImplemented() ;
}
void LensModel::markMask( cv::InputOutputArray r ) {
   throw NotImplemented() ;
}

/* Getters */
cv::Point2d LensModel::getCentre( ) const {
   cv::Point2d xichi =  getXi()/CHI ;
   if ( centredMode ) {
      return tentativeCentre + xichi - getNu() ;
   } else {
      return xichi ;
   }
}
cv::Point2d LensModel::getXi() const { 
   return xi ;
}
double LensModel::getXiAbs() const { 
   cv::Point2d xi = getXi() ;
   return sqrt( xi.x*xi.x + xi.y*xi.y ) ;
}
cv::Point2d LensModel::getTrueXi() const { 
   return CHI*nu ;
}
cv::Point2d LensModel::getNu() const { 
   return nu ;
}
double LensModel::getNuAbs() const { 
   return sqrt( nu.x*nu.x + nu.y*nu.y ) ;
}
cv::Point2d LensModel::getEta() const {
   return eta ;
}
double LensModel::getEtaSquare() const {
   return eta.x*eta.x + eta.y*eta.y ;
}
double LensModel::getEtaAbs() const {
   return sqrt( eta.x*eta.x + eta.y*eta.y ) ;
}
double LensModel::getMaskRadius() const { return 1024*1024 ; }
void LensModel::setNu( cv::Point2d n ) {
   nu = n ;
   xi = nu*CHI ;
   etaOffset = cv::Point2d( 0, 0 ) ;
}
void LensModel::setXi( cv::Point2d x ) {
   if ( rotatedMode ) {
      std::cout << "Alternative viewpoints cannot be supported in rotated mode.\n" ;
      throw NotSupported() ;
   } else {
      throw NotImplemented() ;
   }
}
