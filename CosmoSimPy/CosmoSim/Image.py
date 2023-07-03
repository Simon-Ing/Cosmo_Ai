#! /usr/bin/env python3
# (C) 2022: Hans Georg Schaathun <georg@schaathun.net> 

"""
Post-processing functions for images.
"""

import numpy as np
import cv2

def centreImage(im):
  """
  Shift an image so that the centre of luminence is the centre of the image.

  Input is the image `im` as a numpy array.

  The return value `(centred,(x,y))` consists of the new, shifted
  image `centred` and the co-ordinates `(x,y)` of the ce3ntre of 
  luminence in the original image, written as normal planar co-ordinatres
  where the x-axis points right and the y-axis points up.
  """

  if len(im.shape) > 2:
     grey = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
  else:
      grey = im
  grey = grey.astype(np.float64)
  m,n = grey.shape
  s = grey.sum()

  if s == 0: 
      centred = im
      (xm,ym) = (0,0)
  else:
      xcol = np.array(range(m))[:,np.newaxis]
      yrow = np.array(range(n))[np.newaxis,:]
      xs = xcol*grey
      ys = yrow*grey
      xm = int(round(xs.sum()/s - m/2))
      ym = int(round(ys.sum()/s - n/2))

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
  return (centred,(ym,-xm))

def drawAxes(im):
  m,n = im.shape[:2]
  if m == 0:
      raise Exception( "Image has zero height." )
  if n == 0:
      raise Exception( "Image has zero width." )
  im[int(round(m/2)),:] = 127
  im[:,(round(n/2))] = 127
  return im
