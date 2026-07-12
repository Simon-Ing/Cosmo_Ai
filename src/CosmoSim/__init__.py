# (C) 2022-26: Hans Georg Schaathun <georg@schaathun.net> 

"""
The CosmoSim package comprises several components, including scripts,
GUI, and API.

The root module provides wrappers around most of the C++ classes,
to make the python API more streamlined.
"""

from . import CosmoSimPy as cs
import os

from .CLI import Arguments

from .Dictionary import *

Parameters = Arguments.Parameters
setDebug = cs.setDebug

__version__ = "3.2.3"

def getPathFN(fn):
    """
    Get the absolute path name for file given relative to the location of
    the referencing file.
    """
    dir = os.path.dirname(os.path.abspath(__file__))
    return  os.path.join( dir, fn )

def getMS(minm,maxm=None):
    if minm is None:
        raise RuntimeError( "None argument to getMS()." )
    if maxm is None:
        maxm = minm
        minm = 0
    return [ (m,s) for m in range(minm,maxm+1)
                                    for s in range(1-m%2,m+2,2) ]
