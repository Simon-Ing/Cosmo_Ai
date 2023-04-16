/* (C) 2023: Hans Georg Schaathun <georg@schaathun.net> */

#include "cosmosim/Roulette.h"

#include <symengine/expression.h>
#include <symengine/lambda_double.h>
#include <symengine/parser.h>

#include <thread>
#include <fstream>

RouletteLens::RouletteLens() :
   RouletteAbstractLens::RouletteAbstractLens()
{ 
    std::cout << "Instantiating RouletteLens ... \n" ;
    rotatedMode = false ;
}
RouletteLens::RouletteLens(bool centred) :
   RouletteAbstractLens::RouletteAbstractLens(centred)
{ 
    std::cout << "Instantiating RouletteLens ... \n" ;
    rotatedMode = false ;
}
