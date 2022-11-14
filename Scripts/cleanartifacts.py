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

masked = cv.bitwise_and( im, im, mask=mask )

cv.imwrite( "masked-test.png", masked )
