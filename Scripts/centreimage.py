#! /usr/bin/env python3

"""
Under Construction!

Take an image and return an image with only the 
largest connected component.
"""

import cv2 as cv
import sys
import numpy as np


if len(sys.argv) > 1:
    fn = sys.argv[1] ;
else:
    fn = "image-test.png"
print( "Filename: ", fn )

raw = cv.imread(fn)
print("Raw image Shape: ", raw.shape)

im = cv.cvtColor(raw, cv.COLOR_BGR2GRAY)
print("Shape converted to grey scale: ", im.shape)

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

centred = centreImage(im)
centred = drawAxes(centred)

cv.imwrite( "centred-test.png", centred )
