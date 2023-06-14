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

outcols = [ "index", "filename", "source", "chi", "R", "phi", "einsteinR", "sigma", "sigma2", "theta", "x", "y" ]


def makeSingle(sim,args,name=None,row=None,outstream=None):
    if name == None: name = args.name
    sim.runSim()
    centrepoint = makeOutput(sim,args,name,actual=args.actual,apparent=args.apparent,original=args.original,reflines=args.reflines)


def makeOutput(sim,args,name=None,rot=0,scale=1,actual=False,apparent=False,original=False,reflines=False):
    im = sim.getDistortedImage( 
                    reflines=False,
                    showmask=args.showmask
                ) 

    (cx,cy) = 0,0
    if args.centred:
        (centreIm,(cx,cy)) = centreImage(im)
        if original:
           fn = os.path.join(args.directory,"original-" + str(name) + ".png" ) 
           if reflines: drawAxes(im)
           cv.imwrite(fn,im)
        im = centreIm
    if args.reflines:
        drawAxes(im)

    fn = os.path.join(args.directory, str(name) + ".png" ) 
    cv.imwrite(fn,im)

    if actual:
       fn = os.path.join(args.directory,"actual-" + str(name) + ".png" ) 
       im = sim.getActualImage( reflines=args.reflines )
       cv.imwrite(fn,im)
    if apparent:
       fn = os.path.join(args.directory,"apparent-" + str(name) + ".png" ) 
       im = sim.getApparentImage( reflines=args.reflines )
       cv.imwrite(fn,im)
    return (cx,cy)



if __name__ == "__main__":
    parser = CosmoParser(
          prog = 'CosmoSim makeimage',
          description = 'Generaet an image for given lens and source parameters',
          epilog = '')

    args = parser.parse_args()

    if not args.csvfile:
        raise Exception( "No CSV file given; the --csvfile option is mandatory." )

    print( "Instantiate Simulator ... " )
    if args.amplitudes:
       sim = CosmoSim(fn=args.amplitudes)
    elif args.nterms:
       sim = CosmoSim(maxm=int(args.nterms))
    else:
       sim = CosmoSim()
    print( "Done" )

    if args.nterms:
        sim.setNterms( int(args.nterms) )

    sim.setLensMode( args.lensmode )
    sim.setModelMode( args.modelmode )

    if args.sourcemode:
        sim.setSourceMode( args.sourcemode )
    sim.setSourceParameters( float(args.sigma),
            float(args.sigma2), float(args.theta) )

    if args.imagesize:
        sim.setImageSize( int(args.imagesize) )
        sim.setResolution( int(args.imagesize) )

    sim.setMaskMode( args.mask )

    print( "Load CSV file:", args.csvfile )
    frame = pd.read_csv(args.csvfile)
    cols = frame.columns
    print( "columns:", cols )

    for index,row in frame.iterrows():
            setParameters( sim, row )
            print( "index", row["index"] )
            namestem=row["filename"].split(".")[0]
            makeSingle(sim,args,name=namestem,row=row,outstream=outstream)

    sim.close()
