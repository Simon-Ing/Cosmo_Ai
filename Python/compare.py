#! /usr/bin/env python3

"""
Take an image and return an image with only the 
largest connected component.

This is depecrated as the the new makeimage.py script includes
the post-processing.
"""

import cv2 as cv
import sys
import os
import numpy as np
import argparse

def comparefiles(f1,f2,fout=None):
    im1 = cv.imread(f1).astype(np.float64)
    im2 = cv.imread(f2).astype(np.float64)
    diff = im1 - im2
    print( f"{f1}: "
         + f"Pixel diff: max={np.max(diff)}/min={np.min(diff)}; " 
         + f"Euclidean distance {cv.norm(diff)}" )
    dfim = (diff+256)/2
    if fout != None:
        cv.imwrite( fout, dfim ) 


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
          prog = 'CosmoSim compare',
          description = 'Compare images for debugging purposes',
          epilog = '')

    parser.add_argument('-f', '--files', action='store_true', 
            help="Compare individual files")
    parser.add_argument('dir1')
    parser.add_argument('dir2')
    parser.add_argument('dir3')
    args = parser.parse_args()

    if args.files:
        comparefiles(args.dir1,args.dir2,args.dir3)
    else:
      for fn in os.listdir(args.dir1):
        f1 = os.path.join(args.dir1,fn)
        f2 = os.path.join(args.dir2,fn)
        if args.dir3 == None:
            f3 = None
        else:
            f3 = os.path.join(args.dir3,fn)
        if not os.path.isfile(f1): 
            print( "Not a file: ", f1 )
        elif not os.path.isfile(f2): 
            print( "Not a file: ", f2 )
        else:
          comparefiles(f1,f2,f3)

