/* (C) 2022-23: Hans Georg Schaathun <georg@schaathun.net> */

#include "CosmoSim.h"


namespace py = pybind11;

RouletteSim::RouletteSim() {
   std::cout << "RouletteSim Constructor\n" ;
}


void RouletteSim::initSim( RouletteRegenerator *rsim ) {
   std::cout << "[RouletteSim.cpp] initSim\n" ;
   sim = rsim ;

   return ;
}
void RouletteSim::setImageSize(int sz ) { size = sz ; }

void RouletteSim::setResolution(int sz ) { 
   basesize = sz ; 
   std::cout << "[setResolution] basesize=" << basesize << "; size=" << size << "\n" ;
}

bool RouletteSim::runSim() { 
   std::cout << "[RouletteSim.cpp] runSim() - running similator\n" << std::flush ;
   if ( NULL == sim )
	 throw std::logic_error( "Simulator not initialised" ) ;

   sim->update() ;
   std::cout << "[RouletteSim.cpp] runSim() - complete\n" << std::flush ;
   return true ;
}
