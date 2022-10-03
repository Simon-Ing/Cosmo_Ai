#ifndef SOURCE_H
#define SOURCE_H

#include "opencv2/opencv.hpp"

class Source {

private:
    cv::Mat imgApparent;
    int size ;

protected:
    int drawn ;
public:
    Source(int) ;
    cv::Mat getImage() ;

protected:
    void drawParallel(cv::Mat &img);
    virtual void drawSource(int, int, cv::Mat &) ;
};

class SphericalSource : public Source {

private:
    double sigma ;

public:
    SphericalSource(int,double) ;

protected:
    virtual void drawSource(int, int, cv::Mat &) ;
};

class EllipsoidSource : public Source {

private:
    double sigma1, sigma2, theta ;

protected:
    int size ;

public:
    EllipsoidSource(int,double,double) ;
    EllipsoidSource(int,double,double,double) ;

protected:
    virtual void drawSource(int, int, cv::Mat &) ;

};

#endif //SOURCE_H

