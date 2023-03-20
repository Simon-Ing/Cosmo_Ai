/* (C) 2023: Hans Georg Schaathun <georg@schaathun.net> */

#include "cosmosim/Roulette.h"

#include <symengine/expression.h>
#include <symengine/lambda_double.h>
#include <symengine/parser.h>

#include <thread>
#include <fstream>

RouletteSISLens::RouletteSISLens() :
   SphereLens::SphereLens()
{ 
    std::cout << "Instantiating RouletteSISLens ... \n" ;
    rotatedMode = true ;
}
RouletteSISLens::RouletteSISLens(bool centred) :
   SphereLens::SphereLens(centred)
{ 
    std::cout << "Instantiating RouletteSISLens ... \n" ;
    rotatedMode = true ;
}
RouletteSISLens::RouletteSISLens(std::string fn, bool centred) :
   SphereLens::SphereLens(fn,centred)
{ 
    std::cout << "Instantiating RouletteSISLens ... \n" ;
    rotatedMode = true ;
}
