#! /usr/bin/env python3

"""
Generate an image for given parameters.
"""

import cv2 as cv
import sys
import numpy as np
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
          prog = 'CosmoSim makeimage',
          description = 'Generaet an image for given lens and source parameters',
          epilog = '')

    parser.add_argument('-x', '--x', help="x coordinate")
    parser.add_argument('-y', '--y', help="y coordinate")
    parser.add_argument('-T', '--phi', help="polar coordinate angle (phi)")

    parser.add_argument('-s', '--sigma', default=20, help="source size (sigma)")
    parser.add_argument('-2', '--sigma2', default=10, help="secondary source size (sigma2)")
    parser.add_argument('-t', '--theta', default=45, help="source rotation angle (theta)")

    parser.add_argument('-X', '--chi', default=50, help="lens distance ration (chi)")
    parser.add_argument('-E', '--einsteinradius', default=20, help="Einstein radius")

    parser.add_argument('-n', '--nterms', help="Number of Roulettes terms")
    parser.add_argument('-Z', '--imagesize', help="image size")
    parser.add_argument('-N', '--name', help="simulation name")

    parser.add_argument('-L', '--lensmode', help="lens mode")
    parser.add_argument('-S', '--sourcemode', help="source mode")

    parser.add_argument('-R', '--reflines',action='store_true',
            help="Add reference (axes) lines")
    parser.add_argument('-C', '--centred',action='store_true', help="centre image")

    parser.add_argument('-D', '--directory',help="directory path (for output files)")

    parser.add_argument('-A', '--apparent',action='store_true',help="write apparent image")
    parser.add_argument('-a', '--actual',action='store_true',help="write actual image")

    parser.add_argument('--artifacts',action='store_true',
            help="Attempt to remove artifacts from the Roulettes model")
    parser.add_argument('--sensitivity',
            help="Sensitivity for the connected components (only with -A)")
    args = parser.parse_args()

    sim = CosmoSim()
    if args.phi:
        sim.setPolar( args.x, args.phi )
    else:
        sim.setPolar( args.x, args.y )
    if args.sourcemode:
        sim.setSourceMode( args.sourcemode )
    sim.setSourceParemters( args.sigma, args.sigma2, args.theta )
    if args.lensmode:
        sim.setLensMode( args.lensmode )
    if args.chi:
        sim.setCHI( args.chi/100.0 )
    if args.einsteinradius:
        sim.setEinsteinR( args.einsteinradius/100.0 )
    if args.nterms:
        sim.setNterms( args.nterms )

    sim.setRefLines( args.reflines )

    sim.runSimulator()


    if args.centred:
        print( "centred mode not implemented" )
    print( args.name )
    print( args.directory )
