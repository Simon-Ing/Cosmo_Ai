#include "Simulator.h"
#include <symengine/expression.h>
#include <symengine/lambda_double.h>
#include <thread>
#include <symengine/parser.h>
#include <fstream>

double factorial_(unsigned int n);

PointMassSimulator::PointMassSimulator() :
   Simulator::Simulator()
{ 
   std::cout << "Instantiating Point Mass Simulator ... \n" ;
}
