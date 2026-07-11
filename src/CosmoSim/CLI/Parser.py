# (C) 2026: Hans Georg Schaathun <georg@schaathun.net> 

"""
Evaluate roulette amplitudes using `sympy` in python.

This module has been created for debugging, to validate calculations made in
C++.  It seems to be easier to compute with arbitrary precision in python/sympy
than in C++/symengine.
"""

import sympy
from sympy.parsing.sympy_parser import parse_expr

import numpy as np
import pandas as pd
from .SimImage import getMS

def eval1(expr,x,y,g,precision=64):
    """
    Evaluate the expression `expr` substituting for `x`, `y`, and `g` in 
    high precision.

    This is a sketch.
    """
    x0,y0,g0 = sympy.symbols("x,y,g")
    return expr.evalf( precision, subs={ x0: x, y0: y, g0: g } )


class RouletteParser:
    """ Parse, manage, and evaluate algebraic expressions for roulette amplitudes.
    """
    def __init__(self,fn,g=None,precision=64,verbose=1):
        self.verbose = verbose
        self.precision = precision
        with open(fn, 'r') as f:
            if verbose: print( "Opened file", fn ) 
            c = f.read()
        cs = [ x.split(":") for x in c.split( "\n" ) if x != "" ]
        print( cs[:2] )
        self.alpha = {}
        self.beta = {}
        self.einstein = g
        for x in cs:
            try:
                m,s,a,b = x
                m = int( m )
                s = int( s )
                a = parse_expr( a )
                b = parse_expr( b )
            except:
                print( x )
                raise RuntimeException( "Parse error in amplitudes file." )
            self.alpha[(m,s)] = a
            self.beta[(m,s)] = b
    def getAlpha(self,x,y,g,m,s):
         """get `alpha[m][s]` for Einstein radius `g` at the point (`x`,`y`). """
         return eval1(self.alpha[(m,s)],x,y,g,precision=self.precision)
    def getBeta(self,x,y,g,m,s):
         """get `beta[m][s]` for Einstein radius `g` at the point (`x`,`y`). """
         return eval1(self.beta[(m,s)],x,y,g,precision=self.precision)
    def getAlphaBetas(self,pt,g=None,maxm=5):
        """
        Get the roulette amplitudes for a given point in the source plane.
        """
        if self.verbose>1:
            print ( "[getAlphaBetas] pt=", pt, "in Planar Co-ordinates"  )
        try:
            (x,y) = (pt[0],pt[1])
        except:
            (x,y) = np.array(pt)
        # Scaling is done in getAlpha/getBeta
        if g is None:
            g = self.einstein
        ab1 = { f"alpha[{m}][{s}]" : self.getAlpha(x,y,g,m,s) for (m,s) in getMS(maxm) }
        ab2 = { f"beta[{m}][{s}]" : self.getBeta(x,y,g,m,s) for (m,s) in getMS(maxm) }
        return pd.concat( [ pd.Series( ab1 ), pd.Series( ab2 ) ] )
