#! /usr/bin/env python3

"""
Generate an image for given parameters.
"""

import cv2 as cv
import sys
import os
import numpy as np
import argparse
from CosmoSim.Image import centreImage, drawAxes
from CosmoSim import CosmoSim

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

    parser.add_argument('-L', '--lensmode',
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

    parser.add_argument('-A', '--apparent',action='store_true',help="write apparent image")
    parser.add_argument('-a', '--actual',action='store_true',help="write actual image")

    args = parser.parse_args()

    sim = CosmoSim()
    if args.phi:
        sim.setPolar( int(args.x), int(args.phi) )
    else:
        sim.setXY( int(args.x), int(args.y) )
    if args.sourcemode:
        sim.setSourceMode( args.sourcemode )
    sim.setSourceParameters( int(args.sigma),
            int(args.sigma2), int(args.theta) )
    if args.lensmode:
        sim.setLensMode( args.lensmode )
    if args.chi:
        sim.setCHI( int(args.chi) )
    if args.einsteinradius:
        sim.setEinsteinR( int(args.einsteinradius) )
    if args.imagesize:
        sim.setImageSize( int(args.imagesize) )
    if args.nterms:
        sim.setNterms( int(args.nterms) )

    sim.setMaskMode( args.mask )

    sim.runSim()

    im = sim.getDistortedImage( 
                    reflines=args.reflines,
                    showmask=args.showmask
                ) 

    if args.centred:
        im = centreImage(im)

    fn = os.path.join(args.directory,"image-" + args.name + ".png" ) 
    cv.imwrite(fn,im)

    if args.actual:
       fn = os.path.join(args.directory,"actual-" + args.name + ".png" ) 
       im = sim.getActualImage( reflines=args.reflines )
       cv.imwrite(fn,im)
    if args.apparent:
       fn = os.path.join(args.directory,"apparent-" + args.name + ".png" ) 
       im = sim.getApparentImage( reflines=args.reflines )
       cv.imwrite(fn,im)

    sim.close()
