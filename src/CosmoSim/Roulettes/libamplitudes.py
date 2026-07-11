#! /usr/bin/env python3
# (C) 2023: Hans Georg Schaathun <georg@schaathun.net>

"""
Generate the 50.txt file containing expressions for alpha and beta
for the SIS model in Roulettes.

Usage: `python3 Amplitudes_gen.py [n [nproc]]`

- `n` is the maximum number of terms
- `nproc` is the number of threads
"""

import multiprocessing as mp
import sys
import time
import argparse

import sympy
from sympy import symbols, sqrt, diff, sin, cos, asin, atan2, asinh, pi, log

def identity(f): return f

def listener(fn,q):
    '''Listens for messages on the Queue q and writes to file `fn`. '''
    print( "Listener starts with file ", fn ) 
    with open(fn, 'w') as f:
        print( "Opened file", fn ) 
        cont = True
        while cont:
            # print(f'Jobs running: {}')
            m = q.get()
            # print("got write job:", m)
            if m == 'kill':
                print("Done")
                cont = False
            else:
                f.write(str(m) + '\n')
                f.flush()
        print( "File writer terminated", fn ) 
        f.close()
    if hit_except: print( "Failed to open file ", fn )

def psiSIS():
    print( "psiSIS()" )
    # g is the Einstein radius and (x,y) coordinates in the lens plane
    x, y = symbols('x, y', real=True)
    g = symbols("g", positive=True, real=True)
    psi = g * sqrt(x ** 2 + y ** 2)
    alpha = sympy.factor(diff(psi, x))
    beta = sympy.factor(diff(psi, y))
    return (psi,alpha,beta,x,y)
def psiPM():
    print( "psiPM()" )
    # g is the Einstein radius and (x,y) coordinates in the lens plane
    x, y = symbols('x, y', real=True)
    g = symbols("g", positive=True, real=True)
    psi = g ** 2 * log( sqrt ( x ** 2 + y ** 2 ) )
    alpha = sympy.factor(diff(psi, x))
    beta = sympy.factor(diff(psi, y))
    return (psi,alpha,beta,x,y)
def psiSIE():
    print( "psiSIE()" )
    # g is the Einstein radius and (x,y) coordinates in the lens plane
    x, y = symbols('x, y', real=True)
    g = symbols("g", positive=True, real=True)
    f = symbols("f", positive=True, real=True)
    r = sqrt(x ** 2 + y ** 2)
    alpha = g * sqrt( f/(1-f*f) ) * asinh( ( sqrt( 1-f*f ) / f ) * x/r )
    beta = g * sqrt( f/(1-f*f) ) * asin( sqrt( 1-f*f )* y/r )
    psi = g * sqrt( f/(1-f*f) ) * (
                y * asin( sqrt( 1-f*f )* y/r )
                + x * asinh( ( sqrt( 1-f*f )/f) * x/r ) )
    return (psi,alpha,beta,x,y)


def cfunc(m,k,s):
    p = symbols("phi", real=True)
    f = sin(p)**k*cos(p)**(m-k+1)*cos(s*p)
    return sympy.integrate(f,(p,-pi,+pi))/pi
def sfunc(m,k,s):
    p = symbols("phi", real=True)
    f = sin(p)**k*cos(p)**(m-k+1)*sin(s*p)
    return sympy.integrate(f,(p,-pi,+pi))/pi

def zeroth(lens="SIS"):
    if lens == "SIS":
        return psiSIS()
    if lens == "SIE":
        return psiSIE()
    if lens == "PM":
        return psiPM()
    raise "Unknown lens model"

def ampWrite(alphabeta,f):
    for (m,s) in alphabeta.keys():
            alpha,beta = alphabeta[(m,s)]
            res = f'{m}:{s}:{alpha}:{beta}'
            print ( f'{m}:{s}' )
            f.write(str(res) + '\n')
    f.close()
def ampPrint(alphabeta,fn=None):
    if fn is None:
        ampWrite(alphabeta,sys.stdout)
    else:
        print( "ampPrint" )
        with open(fn, 'w') as f:
            print( "Opened file", fn ) 
            ampWrite(alphabeta,f)


def latex(x): return sympy.latex(sympy.simplify(x))
def texPrint(alphabeta,fn,latex=latex,opt="12pt,paper=a4"):
    print( "texPrint" )
    with open(fn, 'w') as f:
        print( "Opened TeX file", fn ) 
        f.write( "\\documentclass["+opt+"]{scrartcl}\n" )
        f.write( "\\usepackage{amsmath}\n" )
        f.write( "\\usepackage[margin=5mm]{geometry}\n" )
        f.write( "\\begin{document}\n" )
        for (m,s) in alphabeta.keys():

            alpha,beta = alphabeta[(m,s)]
            f.write( f"$$\\alpha_{{{s}}}^{{{m}}} = {latex(alpha)}$$\n" )
            f.write( f"$$\\beta_{{{s}}}^{{{m}}} = {latex(beta)}$$\n" )
        f.write( "\\end{document}\n" )
        f.close()
