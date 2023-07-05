#! /usr/bin/env python3

import cv2 as cv
import sys
import os
import numpy as np
import argparse

def maskedCompare(im,fn):
    print( "maskedCompare", im.shape )
    (m,n) = im.shape[:2]
    cx,cy = int(m/2), int(n/2)
    radius = min(cx,cy)
    for r in range(10,radius,10):
        mask = np.zeros_like(im)
        mask = cv.circle(mask, (cx,cy), r, (1,1,1), -1)
        masked = im*mask
        size = mask.sum()
        norm = cv.norm(masked)
        relnorm = norm/size
        mskim = (masked+256)/2
        print( f"{fn}({r}): {relnorm} ({norm}); size={size}" )
        cv.imwrite( "test.png", mskim )



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
    return { "filename": f1,
             "relnorm": relnorm,
             "norm":    norm,
             "minimum": mn,
             "maximum": mx,
             "image1": im1,
             "image2": im2,
             "diff": diff,
             "outfile": fout }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
          prog = 'CosmoSim compare',
          description = 'Compare images for debugging purposes',
          epilog = '')

    parser.add_argument('-f', '--files', action='store_true', 
            help="Compare individual files")
    parser.add_argument('-t', '--threshold', default=0.001,
            help="Threshold for masked comparison")
    parser.add_argument('-m', '--masked', action='store_true',
            help="Masked comparison")
    parser.add_argument('dir1')
    parser.add_argument('dir2')
    parser.add_argument("-d",'--diff',required=False)
    args = parser.parse_args()

    if args.threshold: 
        threshold = int(args.threshold)
    else:
        threshold = 0.001

    if args.files:
        comparefiles(args.dir1,args.dir2,args.diff)
    else:
      results = []
      maskedresults = []
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
            r = comparefiles(f1,f2,f3) 
            results.append( r )
            if args.masked and r["relnorm"] > threshold:
                assert not f3 is None
                maskedresults.append( maskedCompare( r["diff"], f3 ) )
      badresults = [ r for r in results if r["norm"] > 0 ]
      badresults.sort(key=lambda x : x["relnorm"])
      print( "Bad results sorted by distance:" )
      for r in badresults:
          print( f'{r["filename"]}: {r["relnorm"]} ({r["norm"]}) range {(r["maximum"],r["minimum"])}' )

      if len(badresults) > 0:
          sys.exit(1)
      else:
          sys.exit(0)
