/* (C) 2022: Hans Georg Schaathun <georg@schaathun.net> */

#include "cosmosim/Simulator.h"
#include "sersic.h"
#include <thread>
#include <stdexcept>

EllipsoidSource::EllipsoidSource( int sz, double sig1, double sig2, double thet, LightProfileSpec lightprf, double luminosity, double nsersic) :
        sigma1(sig1),
        sigma2(sig2),
        theta(thet),
        lightprofile(lightprf),
        luminosity(luminosity),
        n_sersic(nsersic),
        Source::Source(sz)
{ if (DEBUG) std::cout << "[EllipsoidSource ] sz=" << sz
                       << "; sigma1=" << sig1
                       << "; sigma2=" << sig2
                       << "; theta=" << thet
                       << "; lightprf=" << lightprf
		       << std::endl ;
}
EllipsoidSource::EllipsoidSource( int sz, double sig1, double sig2, double thet, LightProfileSpec lightprf) :
        sigma1(sig1),
        sigma2(sig2),
        theta(thet),
        lightprofile(lightprf),
        luminosity(15),
        n_sersic(4),
        Source::Source(sz)
{ if (DEBUG) std::cout << "[EllipsoidSource ] sz=" << sz
                       << "; sigma1=" << sig1
                       << "; sigma2=" << sig2
                       << "; theta=" << thet
                       << "; lightprf=" << lightprf
		       << std::endl ;
}

std::string EllipsoidSource::idString() {
   return "Ellipsoid  Source" ;
};

/* Draw the source image.  The sourceSize is interpreted as the standard deviation in a Sersic or Gaussian distribution */
void EllipsoidSource::drawSource(int begin, int end, cv::Mat& dst) {
    for (int row = begin; row < end; row++) {
        for (int col = 0; col < dst.cols; col++) {
            int x = col - dst.cols/2;
            int y = row - dst.rows/2;
            if (lightprofile == LightProfileSpec::CSIM_LIGHT_GAUSSIAN) {
              auto value = (uchar) round (
		    255 * exp( 0.5*(-(x*x)/(sigma1*sigma1) 
			  - (y*y)/(sigma2*sigma2) ) ) );
              dst.at<uchar>(row, col) = value;
            } else if (lightprofile == LightProfileSpec::CSIM_LIGHT_SERSIC) {
	      dst.at<uchar>(row, col) =
                   sersic( n_sersic, luminosity, sigma1, sigma2, x, y ) ;
            }  else {
	       throw std::runtime_error( "Unknown light profile." );
	    }
        }
    }
}

/* drawParallel() split the image into chunks to draw it in parallel using drawSource() */
void EllipsoidSource::drawParallel(cv::Mat& dst){
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
    cv::Mat rot = cv::getRotationMatrix2D( cv::Point(dst.rows/2, dst.cols/2), theta*180/PI, 1);
    cv::warpAffine(dst, dst, rot, dst.size());    // crop distorted image
}

