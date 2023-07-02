#! /usr/bin/env python3

import cv2 as cv
import sys
import os
import numpy as np
import argparse

def comparefiles(f1,f2,fout=None):
    im1 = cv.imread(f1).astype(np.float64)
    im2 = cv.imread(f2).astype(np.float64)
    diff = im1 - im2
    norm = cv.norm(diff)
    relnorm = norm/diff.size
    mx = np.max(diff)
    mn = np.min(diff)
    print( f"{f1}: "
         + f"Pixel diff: max={mx}/min={mn}; " 
         + f"Euclidean distance {norm} or {relnorm} per pizel." )
    dfim = (diff+256)/2
    if fout != None:
        cv.imwrite( fout, dfim ) 
    return (f1,relnorm,norm,mn,mx)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
          prog = 'CosmoSim compare',
          description = 'Compare images for debugging purposes',
          epilog = '')

    parser.add_argument('-f', '--files', action='store_true', 
            help="Compare individual files")
    parser.add_argument('dir1')
    parser.add_argument('dir2')
    parser.add_argument("-d",'--diff',required=False)
    args = parser.parse_args()

    if args.files:
        comparefiles(args.dir1,args.dir2,args.diff)
    else:
      results = []
      for fn in os.listdir(args.dir1):
        f1 = os.path.join(args.dir1,fn)
        f2 = os.path.join(args.dir2,fn)
        if args.diff == None:
            f3 = None
        else:
            f3 = os.path.join(args.diff,fn)
        if not os.path.isfile(f1): 
            print( "Not a file: ", f1 )
        elif not os.path.isfile(f2): 
            print( "Not a file: ", f2 )
        else:
            results.append( comparefiles(f1,f2,f3) )
      badresults = [ r for r in results if r[2] > 0 ]
      badresults.sort(key=lambda x : x[1])
      print( "Bad results sorted by distance:" )
      for r in badresults:
          print( f"{r[0]}: {r[1]} ({r[2]}) range {(r[3],r[4])}" )

