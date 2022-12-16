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

    parser.add_argument('-s', '--sigma', help="source size (sigma)")
    parser.add_argument('-2', '--sigma2', help="secondary source size (sigma2)")
    parser.add_argument('-t', '--theta', help="source rotation angle (theta)")

    parser.add_argument('-X', '--chi', help="lens distance ration (chi)")
    parser.add_argument('-E', '--einstein-radius', help="Einstein radius")

    parser.add_argument('-n', '--nterms', help="Number of Roulettes terms")
    parser.add_argument('-Z', '--image-size', help="image size")
    parser.add_argument('-N', '--name', help="simulation name")

    parser.add_argument('-L', '--lens-mode', help="lens mode")
    parser.add_argument('-S', '--source-mode', help="source mode")

    parser.add_argument('-R', '--reflines',action='store_true',
            help="Add reference (axes) lines")
    parser.add_argument('-C', '--centred',action='store_true', help="centre image")

    parser.add_argument('-D', '--directory',help="directory path (for output files)")

    parser.add_argument('-A', '--apparent-image',help="write apparent image")
    parser.add_argument('-a', '--actual-image',help="write actual image")

    parser.add_argument('--artifacts',action='store_true',
            help="Attempt to remove artifacts from the Roulettes model")
    parser.add_argument('--sensitivity',
            help="Sensitivity for the connected components (only with -A)")
    args = parser.parse_args()

