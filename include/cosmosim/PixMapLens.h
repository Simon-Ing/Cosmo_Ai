#ifndef PIXMAPLENS_H
#define PIXMAPLENS_H

#include "cosmosim/Lens.h"

class PixMapLens : public SampledLens {
public:
    void setPsi( cv::Mat ) ;
    void loadPsi( std::string ) ;
} ;
#endif /* PIXMAPLENS_H */
