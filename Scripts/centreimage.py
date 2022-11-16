#! /usr/bin/env python3

"""
Under Construction!

Take an image and return an image with only the 
largest connected component.
"""

import cv2 as cv
import sys
import numpy as np
import argparse

def centreImage(im):
  m,n = im.shape
  ps = [ (x,y) for x in range(m) for y in range(n) ]
  s = im.sum()
  xs = [ x*im[x,y] for (x,y) in ps ]
  ys = [ y*im[x,y] for (x,y) in ps ]
  xm = int(round(sum(xs)/s - m/2))
  ym = int(round(sum(ys)/s - n/2))
  
  print( f"Centre: ({xm},{ym})\n" )

  centred = np.zeros( (m,n) )
  c1 = np.zeros( (m,n) )

  if xm > 0:
      c1[:-xm,:] = im[xm:,:]
  elif xm < 0:
      c1[-xm:,:] = im[:xm,:]
  else:
      c1 = im
  if ym > 0:
      centred[:,:-ym] = c1[:,ym:]
  elif ym < 0:
      centred[:,-ym:] = c1[:,:ym]
  else:
      centred = c1
  print( f"Range ({centred.min()},{centred.max()})" )
  return centred

def drawAxes(im):
  m,n = im.shape
  im[int(round(m/2)),:] = 127
  im[:,(round(n/2))] = 127
  return im

def cleanImage(im):
   retval, labels, stats, centroids = \
     cv.connectedComponentsWithStats( im, connectivity=8, ltype=cv.CV_32S )


   print( "Labels Shape: ", labels.shape )
   print( "Labels Map: ", labels )
   print( "Labels Range: ", labels.min(), labels.max() )
   print( "Stats: ", stats )

   components = [ x for x in enumerate( stats[:,cv.CC_STAT_AREA] ) ]
   print( "Components: ", components )

   components.sort(reverse=True, key=lambda x : x[1] )
   print( "Components: ", components )

   label = components[1][0]
   mask = ( labels == label ).astype(np.uint8)

   return cv.bitwise_and( im, im, mask=mask )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
          prog = 'CosmoSim centreImage',
          description = 'Centre an image on the centre of light',
          epilog = '')

    parser.add_argument('fn',nargs="?",default="image-test.png") 
    parser.add_argument('outfile',nargs="?",default="centred-test.png") 
    parser.add_argument('-R', '--reflines',
                    action='store_true')  # Add reference lines
    parser.add_argument('-A', '--artifacts',
                    action='store_true')  # Also clean artifacts
    args = parser.parse_args()

    print( "Filename: ", args.fn )

    raw = cv.imread(args.fn)
    print("Raw image Shape: ", raw.shape)

    im = cv.cvtColor(raw, cv.COLOR_BGR2GRAY)
    print("Shape converted to grey scale: ", im.shape)
    print("Image type: ", im.dtype)
    centred = centreImage(im)
    if args.artifacts:
        centred = cleanImage(centred)
    if args.reflines:
        centred = drawAxes(centred)

    cv.imwrite( args.outfile, centred )
