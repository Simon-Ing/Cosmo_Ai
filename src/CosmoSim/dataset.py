#! /usr/bin/env python3
# (C) 2025: Hans Georg Schaathun <georg@schaathun.net>

"""
Generate random sets of lens/source parameters.
The module can be run as a script, taking a TOML file as input
and producing a CSV file as output.
Alternatively, the datasetgen() function can be imported and
used from other modules.
"""

import numpy as np
import tomllib as tl
import numbers

import random 
import argparse
import pandas as pd

from . import Parameters
from .Dictionary import *
from .CLI.Generators import getLens

def isnumeric(x): return isinstance(x, numbers.Number )
def fromPolar(R,phi): 
        x = R*np.cos(np.pi*phi/180)
        y = R*np.sin(np.pi*phi/180)
        return (x,y)

def tomldefaults(toml):
    toml["simulator"] = toml.get( "simulator", {} )
    toml["position"] = toml.get( "position", {} )
    toml["source"] = toml.get( "source", {} )
    toml["lens"] = toml.get( "lens", {} )

def simodels(toml):
    r = toml["simulator"].get( "model", None )
    if r is None:
        r = toml["simulator"].get( "models", None )
    if isinstance(r, str): r = [ r ] 
    return r
def lensmodes(toml):
    r = toml["lens"].get( "mode", None )
    if r is None:
        r = toml["lens"].get( "modes", None )
    if isinstance(r, str): r = [ r ] 
    return r
def configs(toml):
    r = toml["simulator"].get( "config", None )
    if r is None:
        r = toml["simulator"].get( "configs", None )
    if isinstance(r, str): r = [ r ] 
    return r
def srcmode(toml):
    r = toml["source"].get( "mode", [ "e" ] )
    if isinstance(r, str): r = [ r ] 
    return r
def uniform(toml,key,rng=(0,50)):
    if key in toml:
        return toml[key]
    if rng:
        mn = toml.get( f"{key}-min", rng[0] )
        mx = toml.get( f"{key}-max", rng[1] )
        return random.uniform(mn,mx)
    try:
        mn = toml.get( f"{key}-min" )
        mx = toml.get( f"{key}-max" )
        return random.uniform(mn,mx)
    except:
        return None
    
def exponential(toml,key,rng=(0,50),lam=0.8):
    if key in toml:
        return toml[key]
    lambd = toml.get( f"{key}-lambda", lam )
    x = random.expovariate(lambd)
    u = 1 - np.exp(-lambd * x)

    if rng:
        mn = toml.get( f"{key}-min", rng[0] )
        mx = toml.get( f"{key}-max", rng[1] )
        return mn + (mx - mn) * u
    try:
        mn = toml.get( f"{key}-min" )
        mx = toml.get( f"{key}-max" )
        return mn + (mx - mn) * u
    except:
        return None

def rndsource(toml,einstein=50,lens=None,verbose=1):

    src = random.choice( srcmode(toml) )

    sigma = uniform( toml["source"], "sigma", (1,60) )
    sigma2 = uniform( toml["source"], "sigma2", (1,40) )
    theta = uniform( toml["source"], "theta", (0,179) )
    coor = toml["source"].get( "position", "cartesian" )
    n_sersic = uniform(toml["source"], "n_sersic", (1,5))
    luminosity = exponential(toml["source"], "luminosity", (1,20), 2.0)

    if coor == "polar":
        # Polar Source Co-ordinates
        phi = uniform( toml["position"], "phi", (0,359) )
        rmin = toml["position"].get( "r-min" )
        rmax = toml["position"].get( "r-max" )
        if not isnumeric(rmin): rmin = einstein
        if not isnumeric(rmax): rmax = 2.5*rmin

        R =  random.uniform(rmin,rmax)

        # Cartesian Co-ordinates
        (x,y) = fromPolar(R,phi)
    elif coor == "critical":
        phi = uniform( toml["position"], "phi", (0,359) )
        if lens is None:
            raise RuntimeError( "Cannot use critical curve without a known lens." )
        einstein = lens.criticalXi( phi )
        rmin = toml["position"].get( "r-min", 1 )
        rel = toml.get( "r-relativemax", 1.2 )
        R =  random.uniform(rmin,rel*einstein)
        (x,y) = fromPolar(R,phi)
    elif coor == "relative":
        # Polar coordinates relative to the Einstein radius.
        phi = uniform( toml["position"], "phi", (0,359) )
        rmin = toml["position"].get( "r-min", 1 )
        rel = toml.get( "r-relativemax", 1.2 )
        R =  random.uniform(rmin,rel*einstein)
        (x,y) = fromPolar(R,phi)
    else:
        # Cartesian Co-ordinates
        (x,y) = (0,0)
        while (x,y) == (0,0):
            x = uniform( toml["position"], "x", (-100,+100) )
            y = uniform( toml["position"], "y", (-100,+100) )
        R = np.sqrt( x*x + y*y )
        phi = np.atan2(x,y)

    return pd.Series(
        data=[ src, R, phi, sigma, sigma2, theta, n_sersic, luminosity, x, y ],
        index=[ "source", "R", "phi", "sigma", "sigma2", "theta", "n_sersic", "luminosity", "x", "y" ]
        )

def rndlens(toml,verbose=1):
    einsteinradius = uniform( toml["lens"], "einstein", (10,50) )
    orientation = uniform( toml["lens"], "orientation", rng=None )
    ellipseratio = uniform( toml["lens"], "ellipseratio", rng=None )

    return pd.Series(
        data=[  einsteinradius, ellipseratio, orientation ],
        index=[ "einsteinradius", "ellipseratio", "orientation" ]
        )

def lensSpec(toml,verbose=0):
    param = rndlens(toml,verbose)
    eR = param["einsteinradius"]
    c = toml.get( ("cluster","maxrelativelocation"), 1.2 )
    rmax = c*eR
    phi = random.uniform(0,359)
    r = random.uniform(0,rmax)
    (x,y) = fromPolar( r, phi )
    try:
       lens = toml["lens"].get( "mode" )
    except:
        raise RuntimeError( "Lens model undefined" )
    ls = [ lens, x, y, eR ]
    if lens == "SIE":
        ls.extend( [ param["ellipseratio"], param["orientation"] ] )
    elif lens == "SIS":
        pass
    else:
        raise RuntimeError( f"Unknown lens model {lens}" )
    return ls
def lensSpec2String(ls,verbose=0):
    r = [ ls[0] ] + [ str(x) for x in ls[1:] ]
    return "/".join( r )
def lensString(toml,verbose=0):
    return lensSpec2String( lensSpec( toml, verbose=verbose ) )

def getline(toml,idx=0,fn=None,verbose=1):
    """
    Get random parameters for one lensing system, using the distribution defined
    by argument `toml`.  The return value is a pandas Series object, which includes
    the index `idx` and filename `fn` as fields.
    """

    if fn is None: fn = f"image-{idx:06}.png"

    r = pd.Series ( { "index" : idx
                    , "filename" : fn
                    } )

    cfg = simodels(toml) 
    if cfg is not None: r["model"] = random.choice( cfg )
    cfg = configs(toml)
    if cfg is not None: r["config"] = random.choice( cfg )

    nterms = toml["simulator"].get( "nterms", None )
    if nterms is not None: r["nterms"] : nterms


    cluster = toml.get( "cluster", None ) 
    if cluster:
        nc = toml["cluster"].get( "count", 1 )
        if verbose > 1: print( "[getline] Cluster", toml["cluster"] )
        if nc > 1:
            ls0 = [ lensSpec(toml,verbose=verbose) for i in range(nc) ]
            ls1 = [ lensSpec2String( x, verbose=verbose ) for x in ls0 ]
            eRs = [ np.sqrt(x[1]**2+x[2]**2) + x[3] for x in ls0 ]
            eR = max( eRs )
            l = pd.Series( { "cluster" : ";".join( ls1 ) } )
        else:
            raise RuntimeError( "[dataset.py] Singleton or malformed cluster lens" )
        if verbose > 1: print( "[getline] Cluster", l)
    else:
        cfg = lensmodes(toml)
        l =  rndlens( toml, verbose )
        if cfg is not None: l["lens"] = random.choice( cfg )
        eR = l["einsteinradius"]
        if verbose > 1: print( "[getline] Singleton lens", l )
    coor = toml["source"].get( "position", "cartesian" )
    if coor == "critical":
        lcfg = Parameters( { "lens" : dict(l) } )
        if verbose > 1: print( "[getline] Critical positioning", lcfg )
        lens = getLens( lcfg )
        s = rndsource( toml, lens=lens, verbose=verbose )
    else:
        s = rndsource( toml, einstein=eR, verbose=verbose )

    return pd.concat( [ r, l, s ] )

def readtoml(infile,verbose=1):
    """
    Load a TOML file for dataset generation.
    The return value is a nested dict.
    """
    with open(infile, 'rb') as f:
        toml = tl.load(f)
    tomldefaults(toml)
    return toml
def datasetgen(infile,outfile=None,verbose=1):
    toml = readtoml(infile,verbose)
    if verbose > 0:
       print( "[datasetgen]", configs(toml))
       print( "[datasetgen]", srcmode(toml))
    if verbose > 1:
       print(toml)

    n = toml["simulator"].get( "size", 10000 )
    df = pd.DataFrame( [ getline(toml,i+1,verbose=verbose) for i in range(n) ] )
    if outfile:
        if verbose:
            print( "[datasetgen] Output", outfile )
        df.to_csv( outfile, float_format="%.4f", index=False )
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
          prog = 'Dataset generator for CosmoSim',
          description = 'Generate CSV files for the datagen image generator',
          epilog = '')
    parser.add_argument('infile',help="Input file")
    parser.add_argument('outfile',help="Output file")
    args = parser.parse_args()
    datasetgen(args.infile,args.outfile)

