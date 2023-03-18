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

def comparefiles(f1,f2):
    im1 = cv.imread(f1).astype(np.float64)
    im2 = cv.imread(f2).astype(np.float64)
    diff = im1 - im2
    print( f"{f1}: "
         + f"Pixel diff: max={np.max(diff)}/min={np.min(diff)}; " 
         + f"Euclidean distance {cv.norm(diff)}" )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
          prog = 'CosmoSim compare',
          description = 'Compare images for debugging purposes',
          epilog = '')

    parser.add_argument('dir1')
    parser.add_argument('dir2')
    args = parser.parse_args()

    for fn in os.listdir(args.dir1):
        f1 = os.path.join(args.dir1,fn)
        if not os.path.isfile(f1): 
            print( "Not a file: ", f1 )
            break
        f2 = os.path.join(args.dir2,fn)
        if not os.path.isfile(f2): 
            print( "Not a file: ", f2 )
            break
        comparefiles(f1,f2)

