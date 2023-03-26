#ifndef LENS_H
#define LENS_H

class Lens { 

private:

public:
    virtual double psifunction( double, double ) = 0 ;
    virtual double psiXfunction( double, double ) = 0 ;
    virtual double psiYfunction( double, double ) = 0 ;
    virtual void updatePsi() ;
};

class SIS : public Lens { 

private:

public:
    virtual double psifunction( double, double ) = 0 ;
    virtual double psiXfunction( double, double ) = 0 ;
    virtual double psiYfunction( double, double ) = 0 ;
};


#endif // LENS_H
