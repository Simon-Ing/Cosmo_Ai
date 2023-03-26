#ifndef LENS_H
#define LENS_H

class Lens { 

private:
   double einsteinR ;

public:
    virtual double psifunction( double, double ) = 0 ;
    virtual double psiXfunction( double, double ) = 0 ;
    virtual double psiYfunction( double, double ) = 0 ;
    virtual void updatePsi() ;
    virtual void setEinsteinR( double ) ;
};

class SIS : public Lens { 

private:

public:
    virtual double psifunction( double, double ) ;
    virtual double psiXfunction( double, double ) ;
    virtual double psiYfunction( double, double ) ;
};


#endif // LENS_H
