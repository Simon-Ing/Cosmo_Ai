#! /usr/bin/env python3

"""
Generate an image for given parameters.
"""

import cv2 as cv
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

from CosmoSim.Image import drawAxes
from CosmoSim import RouletteSim as CosmoSim,getMSheaders,PsiSpec,ModelSpec

from Arguments import CosmoParser
import pandas as pd

def setParameters(sim,row):
    print( row ) 
    if row.get("source",None) != None:
        sim.setSourceMode( row["source"] )
    if row.get("sigma",None) != None:
        sim.setSourceParameters( row["sigma"],
            row.get("sigma2",-1), row.get("theta",-1) )
    if row.get("imagesize",None) != None:
        sim.setImageSize( row["imagesize"] )
        sim.setResolution( row["imagesize"] )

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

def parseCols(l):
    """Auxiliary function for RouletteAmplitudes."""
    r = [ parseAB(s) for s in l ]
    print( r )
    r = filter( lambda x : x != None, r )
    return r


def makeSingle(sim,args,name=None,row=None):
    print( "makeSingle" )
    sys.stdout.flush()
    if name == None: name = args.name
    sim.runSim()
    print( "runSim() returned" )
    sys.stdout.flush()

    im = sim.getDistortedImage( 
                    reflines=False,
                    showmask=args.showmask
                ) 

    (cx,cy) = 0,0
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

def setAmplitudes( sim, row, coefs ):
    maxm = coefs.getNterms()
    for m in range(maxm+1):
        for s in range((m+1)%2, m+2, 2):
            alpha = row[f"alpha[{m}][{s}]"]
            beta = row[f"beta[{m}][{s}]"]
            print( f"alpha[{m}][{s}] = {alpha}\t\tbeta[{m}][{s}] = {beta}." )
            sim.setAlphaXi( m, s, alpha )
            sim.setBetaXi( m, s, beta )


if __name__ == "__main__":
    print( "[roulettegen.py] Starting ..." )
    parser = CosmoParser(
          prog = 'CosmoSim makeimage',
          description = 'Generaet an image for given lens and source parameters',
          epilog = '')

    args = parser.parse_args()

    if not args.csvfile:
        raise Exception( "No CSV file given; the --csvfile option is mandatory." )

    print( "Instantiate RouletteSim object ... " )
    sim = CosmoSim()
    print( "Done" )

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
            print( "Processing", index )
            sys.stdout.flush()
            sim.setCentre( row["etaX"], row["etaY"] )
            print( "Eta", row["etaX"], row["etaY"] )
            print( "Centre Point", row["centreX"], row["centreY"] )
            sim.initSim() 
            print( "Initialised simulator" )
            sys.stdout.flush()

            setAmplitudes( sim, row, coefs )
            print( "index", row["index"] )
            sys.stdout.flush()
                    
            sim.setSourceParameters( float(row["sigma"]), float(row["sigma2"]),
                                     float(row["theta"]) ) 
            namestem = row["filename"].split(".")[0]
            makeSingle(sim,args,name=namestem,row=row)

    sim.close()
    print( "[roulettegen.py] Done" )
