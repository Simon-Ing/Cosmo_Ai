/* (C) 2022: Hans Georg Schaathun <georg@schaathun.net> */

/* The SphericalSource class implements a a Spherical, Gaussian mass, */

#include "cosmosim/Simulator.h"
#include "cosmosim/Source.h"
#include "sersic.h"

SphericalSource::SphericalSource(int sz, double sig, double idx, double lum, LightProfileSpec lightprf) :
        Source::Source(sz),
        sigma(sig),
        n_sersic(idx),
        luminosity(lum),
        lightprofile(lightprf)
{ 
        if (lightprofile == LightProfileSpec::CSIM_LIGHT_GAUSSIAN) {
	   if (DEBUG>1) std::cout << "[SphericalSource] GAUSS\n" ;
        } else if (lightprofile == LightProfileSpec::CSIM_LIGHT_SERSIC) {
	   if (DEBUG>1) std::cout << "[SphericalSource] SERSIC\n" ;
	}
}

std::string SphericalSource::idString() {
   return "Spherical Source" ;
};

/* Draw the source image.  The sourceSize is interpreted as the standard deviation in a Gaussian distribution */
void SphericalSource::drawSource(int begin, int end, cv::Mat& dst) {
    for (int row = begin; row < end; row++) {
        for (int col = 0; col < dst.cols; col++) {
            int x = col - dst.cols/2;
            int y = row - dst.rows/2;
            if (lightprofile == LightProfileSpec::CSIM_LIGHT_GAUSSIAN) {
                auto value = (uchar)round(255 * exp((-x * x - y * y) / (2.0*sigma*sigma)));
                dst.at<uchar>(row, col) = (uchar)value;
            } else if (lightprofile == LightProfileSpec::CSIM_LIGHT_SERSIC) {
	       dst.at<uchar>(row, col) =
                   sersic( n_sersic, luminosity, sigma, sigma, x, y ) ;
            }
        }
    }
}

