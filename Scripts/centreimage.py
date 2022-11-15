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

m,n = im.shape
ps = [ (x,y) for x in range(m) for y in range(n) ] )
s = im.sum()
xs = [ x*im[x,y] for (x,y) in ps ]
ys = [ y*im[x,y] for (x,y) in ps ]
xm = sum(xs)/s
ym = sum(ys)/s

print( f"Centre: ({xm},{ym})\n" )

# cv.imwrite( "centred-test.png", masked )
