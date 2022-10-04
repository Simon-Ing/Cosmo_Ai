/* (C) 2022: Hans Georg Schaathun <hg@schaathun.net> */

/* The Source class implements the source model for the gravitational lensing simulator.
 * Currently it implements a Spherical, Gaussian mass,
 * It should be turned into an abstract class, with subclasses for each
 * of a range of source models.
 */

#include "Source.h"
#include <thread>

Source::Source(int sz) :
        size(sz)
{ 
    imgApparent = cv::Mat(size, size, CV_8UC3, cv::Scalar(0, 0, 0)) ;
    drawn = 0 ;
}

/* Getters for the images */
cv::Mat Source::getImage() { 
   if ( ! drawn ) {
      drawParallel( imgApparent ) ;
      drawn = 1 ;
   }
   return imgApparent ; 
}

/* drawParallel() split the image into chunks to draw it in parallel using drawSource() */
void Source::drawParallel(cv::Mat& dst){
    int n_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads_vec;
    for (int i = 0; i < n_threads; i++) {
        int begin = dst.rows / n_threads * i;
        int end = dst.rows / n_threads * (i + 1);
        std::thread t([begin, end, &dst, this]() { this->drawSource(begin, end, dst ); });
        threads_vec.push_back(std::move(t));
    }
    for (auto& thread : threads_vec) {
        thread.join();
    }
}

