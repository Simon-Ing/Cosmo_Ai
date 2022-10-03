#ifndef SOURCE_H
#define SOURCE_H

#include "opencv2/opencv.hpp"

class Source {

private:
    cv::Mat imgApparent;
    int size ;

public:
    Source(int) ;
    cv::Mat getImage() ;

private:
    void drawParallel(cv::Mat &img);
    virtual void drawSource(int begin, int end, cv::Mat &img) ;
};

class SphericalSource : public Source {

private:
    double sigma ;

public:
    SphericalSource(int,double) ;

private:
    virtual void drawSource(int begin, int end, cv::Mat &img);
};

class EllipsoidSource : public Source {

private:
    double sigma1, sigma2, theta ;

protected:
    int size ;

public:
    EllipsoidSource(int,double,double) ;
    EllipsoidSource(int,double,double,double) ;

private:
    virtual void drawSource(int begin, int end, cv::Mat &img);

};

#endif //SOURCE_H

