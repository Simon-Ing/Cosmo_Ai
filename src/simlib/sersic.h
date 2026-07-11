/* (C) 2026: Hans Georg Schaathun <georg@schaathun.net> */

#define pi 3.141592

inline double sersic( double n_sersic, double luminosity, 
               double sigma1, double sigma2, 
               double x, double y ) {
   auto q = sigma2/sigma1;  // Ellipticity
   auto r = std::sqrt( (x/q)*(x/q) + y*y ); // radius

   auto bn = 1.992*n_sersic - 0.3271;
   auto L = luminosity * 1000 ;

   auto re = sigma1 ; // effective radius

   auto I_eff = L*std::pow(bn, 2.0 * n_sersic)
              / (2 * pi * sigma1*sigma2*n_sersic * std::exp(bn)
              * std::tgamma(2.0*n_sersic)) ;

   auto value = round(I_eff*std::exp(-bn*((std::pow(r/re, 1.0/n_sersic))-1.0)));

   if (value > 255) { value = 255; }

   return value ;
}
