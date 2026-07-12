#! /usr/bin/env python3
# (C) 2022-26: Hans Georg Schaathun <georg@schaathun.net> 

"""
Helper functions for images including post-processing and viewing.

The viewing functions are adapted primarily for use in Jupyter
Notebook and may or may not work in other use cases.
"""

import numpy as np
import cv2
import argparse
import matplotlib.pyplot as plt

def imshow(im,title=None):
    """
    Show a greyscale image with an optional title.
    This works directly in Juyter Lab.  
    """
    plt.imshow(im,cmap="grey", vmin=0, vmax=255)
    if title is not None:
        plt.title( title )
    plt.axis( "off" )

def showImages(ims,size=(1,3),titles=None):
    """
    Show several images as subfigures.  The size argument is the number of images,
    as (rows,columns).
    """
    (x,y) = size
    fig = plt.figure(figsize=(5*y, 5*x))
    fig.tight_layout(pad=0.0)
    if titles is None:
        titles = [ None for _ in ims ]
    for (i,(im,t)) in enumerate(zip(ims,titles)):
        fig.add_subplot(x, y, i+1)
        imshow(im,t)

def imageCompare(im1,im2,title1=None,title2=None,axiscross=False):
    """
    Plot two images plus their difference image.
    If `axiscross` is true, an axis cross marking the centre of the image
    is added to the two original images, but not to the difference image.
    """
    diff = imageDiff(im1,im2)
    if axiscross:
        im1 = im1.copy()
        im2 = im2.copy()
        drawAxes( im1 )
        drawAxes( im2 )
    return showImages( [ im1, diff, im2 ]
               , size=(1,3)
               , titles=[ title1, "Difference", title2 ] )

def imageCoordinate( pt, im ):
    """
    Translate a Cartesian point to image co-ordinates in the given image.
    Only the size of the image is relevant.  The origin is the centre of the image.
    """
    nrows, ncols = im.shape[:2]
    (x,y) = pt
    return (  round(x + ncols/2), round(nrows/2 - y) )

def annotateCircle(im,pt,radius,colour=(0,0,255)):
    """
    Mark a point on an image.
    This is crude and should be extended with different shapes as
    well as text.
    """
    if len(im.shape) == 2:
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    pt = imageCoordinate( pt, im )
    print( "Annotation at", pt, "colour", colour )
    im = cv2.circle(im,center=pt,radius=round(radius),color=colour, thickness=1)
    return im
def annotatePoint(im,pt,colour=(0,0,255)):
    """
    Mark a point on an image.
    This is crude and should be extended with different shapes as
    well as text.
    """
    if len(im.shape) == 2:
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    r = 4
    pt = imageCoordinate( pt, im )
    print( "Annotation at", pt, "colour", colour )
    im = cv2.circle(im,center=pt,radius=r,color=colour, thickness=1)
    im = cv2.circle(im,center=pt,radius=1,color=colour, thickness=-1)
    return im

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
      pt = (0,0)
  else:
      xcol = np.array(range(m))[:,np.newaxis]
      yrow = np.array(range(n))[np.newaxis,:]
      xs = xcol*grey
      ys = yrow*grey
      xm = xs.sum()/s - m/2
      ym = ys.sum()/s - n/2
      pt = (ym,-xm)
      centred = translateImage(im,pt)

  return (centred,pt)

def translateImage(im,pt):
    """Translate the image so that `pt` becomes the new centre.
    The point `pt` is given as Cartesian coordinates relative to the
    centre of the image.
    """
    (ym,xm) = pt
    R = np.float32( [ [ 1, 0, -ym ], [ 0, 1, xm ] ] )
    if len(im.shape) > 2:
        m,n,_ = im.shape
    else:
        m,n = im.shape
    return cv2.warpAffine(im,R,(n,m))

def crop(im,cropsize=256,verbose=1):
    """
    Crop the image `im` to size cropsize x cropsize, keeping the centre of
    the image.
    """
    if verbose: print( f"[crop] cropsize={cropsize}" )
    (m,n) = im.shape[:2]
    if cropsize < min(m,n):
        assert m == n
        c = (m-cropsize)/2
        c1 = int(np.floor(c))
        c2 = int(np.ceil(c))
        if len(im.shape) == 2:
            im = im[c1:-c2,c1:-c2]
        else:
            im = im[c1:-c2,c1:-c2,:]
        assert cropsize == im.shape[0]
        assert cropsize == im.shape[1]
    return im

def drawAxes(im):
    """Draw an axis cross on the `im` marking the centre of the image."""
    m,n = im.shape[:2]
    if m == 0:
        raise Exception( "Image has zero height." )
    if n == 0:
        raise Exception( "Image has zero width." )
    im[int(round(m/2)),:] = 127
    im[:,(round(n/2))] = 127
    return im

def overlay(im1,im2,alpha=1,beta=1,gamma=0):
    d1 = im1.shape
    d2 = im2.shape
    if len(d1) != len(d2):
        if len(d1) == 2:
            im1 = cv2.cvtColor( im1, cv2.COLOR_GRAY2RGB )
        if len(d2) == 2:
            im2 = cv2.cvtColor( im2, cv2.COLOR_GRAY2RGB )
    return cv2.addWeighted(im1, alpha, im2, beta, gamma)

def imageDiff(im1,im2):
    """
    Return the difference image of im1 and im2, scaling pixel values so that 
    zero difference becomes uniform greyscale.
    """
    return ( (im1.astype(float) - im2.astype(float) + 256)/2 ).astype(np.uint8)


if __name__ == "__main__":
    parser = argparse.ArgumentParser( prog = 'Test Image Centring')
    parser.add_argument('-i', '--infile', default="test.png", help="Input file")
    parser.add_argument('-o', '--outfile', default="test-centre.png", help="Output file for centred file.")
    parser.add_argument('-R', '--withaxes', default="test-axes.png", help="Output file with axes")
    args = parser.parse_args()
    im = cv2.imread( args.infile )
    im1 = centreImage(im)[0]
    drawAxes(im)
    drawAxes(im1)
    cv2.imwrite( args.outfile, im1 )
    cv2.imwrite( args.withaxes, im )

