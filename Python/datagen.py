#! /usr/bin/env python3

"""
Generate an image for given parameters.
"""

import cv2 as cv
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

import argparse
from CosmoSim.Image import centreImage, drawAxes
from CosmoSim import CosmoSim,getMSheaders

import pandas as pd

outcols = [ "index", "filename", "source", "chi", "sigma", "sigma2", "theta" ]

def setParameters(sim,row):
    print( row ) 
    if row.get("y",None) != None:
        print( "XY", row["x"], row["y"] )
        sim.setXY( row["x"], row["y"] )
    elif row.get("phi",None) != None:
        print( "Polar", row["x"], row["phi"] )
        sim.setPolar( row["x"], row["phi"] )
    if row.get("source",None) != None:
        sim.setSourceMode( row["source"] )
    if row.get("sigma",None) != None:
        sim.setSourceParameters( row["sigma"],
            row.get("sigma2",-1), row.get("theta",-1) )
    if row.get("lens",None) != None:
        sim.setModelMode( row["lens"] )
    if row.get("chi",None) != None:
        sim.setCHI( row["chi"] )
    if row.get("einsteinR",None) != None:
        sim.setEinsteinR( row["einsteinR"] )
    if row.get("imagesize",None) != None:
        sim.setImageSize( row["imagesize"] )
        sim.setResolution( row["imagesize"] )
    if row.get("nterms",None) != None:
        sim.setNterms( row["nterms"] )

def makeSingle(sim,args,name=None,row=None,outstream=None):
    if name == None: name = args.name
    sim.runSim()
    centrepoint = makeOutput(sim,args,name,actual=args.actual,apparent=args.apparent)
    if args.join:
        # sim.setMaskMode(False)
        sim.runSim()
        sim.maskImage(float(args.maskscale))
        joinim = sim.getDistorted(False)
        # joinim = sim.getDistortedImage(False)
        nc = int(args.components)
        for i in range(1,nc):
           sim.moveSim(rot=2*i*np.pi/nc,scale=1)
           sim.maskImage(float(args.maskscale))
           im = sim.getDistorted(False)
           # im = sim.getDistortedImage(False)
           joinim = np.maximum(joinim,im)
        fn = os.path.join(args.directory,"join-" + str(name) + ".png" ) 
        if args.reflines:
            drawAxes(joinim)
        cv.imwrite(fn,joinim)
    if args.family:
        sim.moveSim(rot=-np.pi/4,scale=1)
        makeOutput(sim,args,name=f"{name}-45+1")
        sim.moveSim(rot=+np.pi/4,scale=1)
        makeOutput(sim,args,name=f"{name}+45+1")
        sim.moveSim(rot=0,scale=-1)
        makeOutput(sim,args,name=f"{name}+0-1")
        sim.moveSim(rot=0,scale=2)
        makeOutput(sim,args,name=f"{name}+0+2")
    if args.psiplot:
        a = sim.getPsiMap()
        print(a.shape, a.dtype)
        print(a)
        nx,ny = a.shape
        X, Y = np.meshgrid( range(nx), range(ny) )
        hf = plt.figure()
        ha = hf.add_subplot(111, projection='3d')
        ha.plot_surface(X, Y, a)
        fn = os.path.join(args.directory,"psi-" + str(name) + ".svg" ) 
        plt.savefig( fn )
        plt.close()
    if args.kappaplot:
        a = sim.getMassMap()
        print(a.shape, a.dtype)
        print(a)
        nx,ny = a.shape
        X, Y = np.meshgrid( range(nx), range(ny) )
        hf = plt.figure()
        ha = hf.add_subplot(111, projection='3d')
        ha.plot_surface(X, Y, a)
        fn = os.path.join(args.directory,"kappa-" + str(name) + ".svg" ) 
        plt.savefig( fn )
        plt.close()
    if outstream:
        maxm = int(args.nterms)
        ab = sim.getAlphaBetas(maxm,pt=centrepoint)
        r = [ row[x] for x in outcols ]
        print(r)
        r.append( centrepoint[0] )
        r.append( centrepoint[1] )
        print(ab)
        r += ab
        line = ",".join( [ str(x) for x in r ] )
        line += "\n"
        outstream.write( line )


def makeOutput(sim,args,name=None,rot=0,scale=1,actual=False,apparent=False):
    im = sim.getDistortedImage( 
                    reflines=False,
                    showmask=args.showmask
                ) 

    (cx,cy) = 0,0
    if args.centred:
        (im,(cx,cy)) = centreImage(im)
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
    parser = argparse.ArgumentParser(
          prog = 'CosmoSim makeimage',
          description = 'Generaet an image for given lens and source parameters',
          epilog = '')

    parser.add_argument('-x', '--x', default=0, help="x coordinate")
    parser.add_argument('-y', '--y', default=0, help="y coordinate")
    parser.add_argument('-T', '--phi', help="polar coordinate angle (phi)")

    parser.add_argument('-s', '--sigma', default=20, help="source size (sigma)")
    parser.add_argument('-2', '--sigma2', default=10, help="secondary source size (sigma2)")
    parser.add_argument('-t', '--theta', default=45, help="source rotation angle (theta)")

    parser.add_argument('-X', '--chi', default=50, help="lens distance ration (chi)")
    parser.add_argument('-E', '--einsteinradius', default=20, help="Einstein radius")

    parser.add_argument('-n', '--nterms', default=10, help="Number of Roulettes terms")
    parser.add_argument('-Z', '--imagesize', default=400, help="image size")

    parser.add_argument('-l', '--lensmode',
            default="SIS", help="lens mode")
    parser.add_argument('-L', '--modelmode',
            default="Point Mass (exact)", help="lens mode")
    parser.add_argument('-S', '--sourcemode',
            default="Spherical", help="source mode")

    parser.add_argument('-R', '--reflines',action='store_true',
            help="Add reference (axes) lines")
    parser.add_argument('-C', '--centred',action='store_true', help="centre image")
    parser.add_argument('-M', '--mask',action='store_true',
            help="Mask out the convergence circle")
    parser.add_argument('-m', '--showmask',action='store_true',
            help="Mark the convergence circle")

    parser.add_argument('-N', '--name', default="test",
            help="simulation name")
    parser.add_argument('-D', '--directory',default="./",
            help="directory path (for output files)")
    parser.add_argument('-O', '--maskscale',default="0.9",
            help="Scaling factor for the mask radius")
    parser.add_argument('-c', '--components',default="6",
            help="Number of components for joined image")

    parser.add_argument('-P', '--psiplot',action='store_true',default=False,
            help="Plot lens potential as 3D surface")
    parser.add_argument('-K', '--kappaplot',action='store_true',default=False,
            help="Plot mass distribution as 3D surface")

    parser.add_argument('-f', '--family',action='store_true',
            help="Several images moving the viewpoint")
    parser.add_argument('-J', '--join',action='store_true',
            help="Join several images from different viewpoints")
    parser.add_argument('-F', '--amplitudes',help="Amplitudes file")
    parser.add_argument('-A', '--apparent',action='store_true',help="write apparent image")
    parser.add_argument('-a', '--actual',action='store_true',help="write actual image")
    parser.add_argument('-o', '--outfile',
            help="Output CSV file")
    parser.add_argument('-i', '--csvfile',
            help="Dataset to generate (CSV file)")

    args = parser.parse_args()

    print( "Instantiate Simulator ... " )
    if args.amplitudes:
       sim = CosmoSim(fn=args.amplitudes)
    elif args.nterms:
       sim = CosmoSim(maxm=int(args.nterms))
    else:
       sim = CosmoSim()
    print( "Done" )
    if args.phi:
        sim.setPolar( float(args.x), float(args.phi) )
    else:
        sim.setXY( float(args.x), float(args.y) )
    if args.sourcemode:
        sim.setSourceMode( args.sourcemode )
    sim.setSourceParameters( float(args.sigma),
            float(args.sigma2), float(args.theta) )
    if args.lensmode:
        sim.setLensMode( args.lensmode )
    if args.modelmode:
        sim.setModelMode( args.modelmode )
    if args.chi:
        sim.setCHI( float(args.chi) )
    if args.einsteinradius:
        sim.setEinsteinR( float(args.einsteinradius) )
    if args.imagesize:
        sim.setImageSize( int(args.imagesize) )
        sim.setResolution( int(args.imagesize) )
    if args.nterms:
        sim.setNterms( int(args.nterms) )
    if args.outfile:
        outstream = open(args.outfile,"wt")
        headers = ",".join( outcols + [ "x", "y" ] + getMSheaders(int(args.nterms)) )
        headers += "\n"
        outstream.write(headers)
    else:
        outstream = None

    sim.setMaskMode( args.mask )

    if args.csvfile:
        print( "Load CSV file:", args.csvfile )
        frame = pd.read_csv(args.csvfile)
        cols = frame.columns
        print( "columns:", cols )
        for index,row in frame.iterrows():
            setParameters( sim, row )
            print( "index", row["index"] )
            namestem=row["filename"].split(".")[0]
            makeSingle(sim,args,name=namestem,row=row,outstream=outstream)
    else:
        makeSingle(sim,args)
    sim.close()
    if outstream != None: outstream.close()
