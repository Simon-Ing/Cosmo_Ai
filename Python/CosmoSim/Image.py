#! /usr/bin/env python3
# (C) 2022: Hans Georg Schaathun <georg@schaathun.net> 

"""
Post-processing functions for images.
"""

import numpy as np
import cv2

def centreImage(im):

  if len(im.shape) > 2:
     grey = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
  else:
      grey = im
  m,n = grey.shape
  ps = [ (x,y) for x in range(m) for y in range(n) ]
  s = grey.sum()
  if s == 0:
      (xm,ym) = (0,0)
  else:
      xs = [ np.sum(x*grey[x,y]) for (x,y) in ps ]
      ys = [ np.sum(y*grey[x,y]) for (x,y) in ps ]
      xm = int(round(sum(xs)/s - m/2))
      ym = int(round(sum(ys)/s - n/2))

  centred = np.zeros( im.shape )
  c1 = np.zeros( im.shape )

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
  print( f"Centre: ({xm},{ym}) ;  " 
       + f"Range ({centred.min()},{centred.max()})" )
  return (centred,(xm,ym))

def drawAxes(im):
  m,n = im.shape[:2]
  im[int(round(m/2)),:] = 127
  im[:,(round(n/2))] = 127
  return im
