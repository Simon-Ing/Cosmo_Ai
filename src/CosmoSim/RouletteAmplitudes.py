#! /usr/bin/env python3
# (C) 2023-24: Hans Georg Schaathun <georg@schaathun.net>

"""
Class to handle datasets of roulette amplitudes.
"""

class RouletteAmplitudes:
    """Parse the CSV headers to find which amplitudes are defined in the file.
    Making it a class may be excessive, but done in case we need other information
    extracted from the process.
    """
    def __init__(self,s):
        self.coeflist = parseCols(s)
        self.maxm = max( [ m for (_,_,(m,_)) in self.coeflist ] )
    def getNterms(self): return self.maxm

def parseAB(s):
    """Auxiliary function for the RouletteAmplitudes class."""
    a = s.split("[")
    if len(a) < 2:
        return None
    elif not a[0] in [ "alpha", "beta" ]:
        return None
    elif len(a) == 2:
       a, bracket = a
       idxstring, = bracket.split("]")
       l = [ int(i) for i in idxstring.split(",") ]
    elif len(a) == 3:
       l = [ int(x.split("]")[0]) for x in a[1:] ]
       a = a[0]
    else:
        return None
    return (a,s,tuple(l))

def parseCols(l,verbose=0):
    """Auxiliary function for the RouletteAmplitudes class."""
    r = [ parseAB(s) for s in l ]
    if verbose: print( r )
    r = filter( lambda x : x != None, r )
    return r

