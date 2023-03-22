#! /usr/bin/env python3
# (C) 2022: Hans Georg Schaathun <georg@schaathun.net> 

"""
Post-processing functions for images.
"""

import numpy as np

def centreImage(im):
  m,n = im.shape[:2]
  ps = [ (x,y) for x in range(m) for y in range(n) ]
  s = im.sum()
  xs = [ np.sum(x*im[x,y]) for (x,y) in ps ]
  ys = [ np.sum(y*im[x,y]) for (x,y) in ps ]
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
  return centred

def drawAxes(im):
  m,n = im.shape[:2]
  im[int(round(m/2)),:] = 127
  im[:,(round(n/2))] = 127
  return im
