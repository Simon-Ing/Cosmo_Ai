#! /usr/bin/env python3

"""
Generate an image for given parameters.
"""

import cv2 as cv
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

from CosmoSim.Image import centreImage, drawAxes
from CosmoSim import CosmoSim,getMSheaders

from Arguments import CosmoParser, setParameters
import pandas as pd

outcols = [ "index", "filename", "source", "chi", "R", "phi", "einsteinR",
            "sigma", "sigma2", "theta", "x", "y" ]

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
    """Auxiliary function for RouletteAmplitudes."""
    a = t.split("[")
    if len(a) == 0:
        return None
    elif not a in [ "alpha", "beta" ]:
        return None
    a, tt = a
    idxstring, = tt.split("]")
    l = [ int(i) for i in idx.split(",") ]
    return (a,s,tuple(l))

def parseCols(l):
    """Auxiliary function for RouletteAmplitudes."""
    r = [ parseAB(s) for s in l ]
    r = filter( lambda x : return x != None, r )
    return r


def makeSingle(sim,args,name=None,row=None,outstream=None):
    if name == None: name = args.name
    sim.runSim()

    im = sim.getDistortedImage( 
                    reflines=False,
                    showmask=args.showmask
                ) 

    (cx,cy) = 0,0
    if args.centred:
        (centreIm,(cx,cy)) = centreImage(im)
        if args.original:
           fn = os.path.join(args.directory,"original-" + str(name) + ".png" ) 
           if reflines: drawAxes(im)
           cv.imwrite(fn,im)
        im = centreIm
    if args.reflines:
        drawAxes(im)

    fn = os.path.join(args.directory, str(name) + ".png" ) 
    cv.imwrite(fn,im)

    if args.actual:
       fn = os.path.join(args.directory,"actual-" + str(name) + ".png" ) 
       im = sim.getActualImage( reflines=args.reflines )
       cv.imwrite(fn,im)
    if args.apparent:
       fn = os.path.join(args.directory,"apparent-" + str(name) + ".png" ) 
       im = sim.getApparentImage( reflines=args.reflines )
       cv.imwrite(fn,im)
    return (cx,cy)

def setAmplitudes( sim, row, coefs ): pass


if __name__ == "__main__":
    parser = CosmoParser(
          prog = 'CosmoSim makeimage',
          description = 'Generaet an image for given lens and source parameters',
          epilog = '')

    args = parser.parse_args()

    if not args.csvfile:
        raise Exception( "No CSV file given; the --csvfile option is mandatory." )

    print( "Instantiate Simulator ... " )
    sim = CosmoSim()
    print( "Done" )

    # sim.setLensMode( args.lensmode )
    # sim.setModelMode( args.modelmode )

    if args.sourcemode:
        sim.setSourceMode( args.sourcemode )

    if args.imagesize:
        sim.setImageSize( int(args.imagesize) )
        sim.setResolution( int(args.imagesize) )

    sim.setMaskMode( args.mask )

    print( "Load CSV file:", args.csvfile )
    frame = pd.read_csv(args.csvfile)
    cols = frame.columns
    print( "columns:", cols )
    
    coefs = RouletteAmplitudes(cols)
    sim.setNterms( coefs.getNterms() )
    print( "Number of roulette terms: ", coefs.getNterms() )

    for index,row in frame.iterrows():
            setAmplitudes( sim, row, coefs )
            print( "index", row["index"] )
            sim.setSourceParameters( float(row["sigma"], float(row["sigma2"], 
                                     float(row["theta"] ) 
            namestem=row["filename"].split(".")[0]
            makeSingle(sim,args,name=namestem,row=row,outstream=outstream)

    sim.close()
