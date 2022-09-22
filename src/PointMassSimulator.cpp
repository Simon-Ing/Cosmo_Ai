/* (C) 2022: Hans Georg Schaathun <hg@schaathun.net> */
/* The point mass model is included in Simulator as a default *
 * implementation, and this this class overrides nothing.     */

#include "Simulator.h"
#include <symengine/expression.h>
#include <symengine/lambda_double.h>
#include <thread>
#include <symengine/parser.h>
#include <fstream>

double factorial_(unsigned int n);

