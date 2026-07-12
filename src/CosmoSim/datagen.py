#! /usr/bin/env python3
# (C) 2024: Hans Georg Schaathun <georg@schaathun.net>

"""
Generate an image for given parameters.
"""

import tomllib as tl

import sys
import os
import matplotlib.pyplot as plt
from .dataset import datasetgen

from .Image import centreImage, drawAxes, crop, annotatePoint, annotateCircle, translateImage

from .CLI.Arguments import Parameters
from .CLI.SimImage import SimImage

import pandas as pd

def makeSingle(param=None,outcols=None,verbose=0):
    """Process a single parameter set, given either as a pandas row or
    just as args parsed from the command line.
    """
    if param is None: param = Parameters()
    if verbose: print( f"[makeSingle] verbose={verbose}" )
    imsim = SimImage(param=param,outcols=outcols,verbose=verbose)
    imsim.saveImage()
    if param.get( "actual" ): imsim.getActual()
    if verbose: print( "makeSingle() returns" )
    return imsim

def datagen(args,param=None):

    if args.rnd:
        frame = datasetgen(args.toml,args.csvfile,verbose=args.verbose)
    elif not args.csvfile:
        raise RuntimeError( "CSV file needed for batch mode" )
    else:
        print( "Load CSV file:", args.csvfile )
        frame = pd.read_csv(args.csvfile)

    outcols = list(frame.columns)
    print( "columns:", outcols )
    dfs = []
    for index,row in frame.iterrows():
        if args.verbose:
            print( "[datagen] Processing", index )
        param.setRow( row )
        imsim = makeSingle(param,outcols=outcols,verbose=args.verbose)
        if args.outfile:
            dfs.append( imsim.getData() )
    df = pd.DataFrame( dfs )
    if args.outfile:
           if args.mldata:
               dropcol = [ "index", "source", "config", "nterms",
                         "centreX", "centreY", "reletaX", "reletaY",
                         "offsetX", "offsetY" ]
               print( df.columns )
               for c in dropcol:
                   try:
                       df.drop( columns=c, inplace=True )
                   except:
                       print( "No column", c )
           df.to_csv(args.outfile, sep=",", index=False)
    print( "simulator done" )

if __name__ == "__main__":
    sys.exit( "[CosmoSim] datagen - deprecated." )
