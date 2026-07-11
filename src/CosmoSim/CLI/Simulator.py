#! /usr/bin/env python3
# (C) 2026: Hans Georg Schaathun <georg@schaathun.net>

"""
Abstract simulator class to manage bulk image generation.

Different subclasses are required for different backend simulators, like
`RouletteRegenerator` and `CosmoSim`.
The `GenericSim` superclass contains shared methods.

Subclasses shoulded override the constructor, calling the superclass constructor
first and `initSim()` last.  In the middle, the `self.sim` must be set with a
backend simulator.

The `initSim` method reconfigures the backend simulator, and generates the 
distorted image.
Since the object retains much of the simulation data, extensive information
and annotated images may be retrieved as well.
"""

import cv2 as cv
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from .. import setDebug
from ..Image import centreImage, drawAxes, crop, annotatePoint, annotateCircle, translateImage
from .Arguments import CosmoParser, Parameters

class GenericSim:
    """
    This is a generic superclass for shared methods.
    """
    def __init__(self,param=None,outcols=None,verbose=1):
        """
        Normally the constructor should be overridden, calling the superclass
        constructor first and `initSim()` last.
        """
        if verbose is None: verbose=1
        self.verbose = verbose
        setDebug( verbose )
        if verbose > 1: print( f"[GenericSim] init (verbose={verbose}) ..." )
        sys.stdout.flush()

        self.outcols = outcols

        if param is None: 
            if verbose: print( "[GenericSim] No parameters given. Using defaults" )
            param = Parameters()
        self.param = param
        if self.verbose > 2:
            print( "[GenericSim]", self.param )

        if self.verbose > 2:
            print( "[GenericSim] xireference =", self.param.get( "xireference" ) )
    def runSim(self):
        """
        Run the simulator in the present state.
        """
        self.sim.update()
        self.image = self.getDistortedImage( )

        (self.centreimage,self.centrepoint) = centreImage(self.image)
        if self.verbose: 
            print( "[Simulator] Centre Point",
                f"({self.centrepoint[0]:.2f},{self.centrepoint[1]:.2f})",
                "(Centre of Luminence in Planar Co-ordinates)" )

    def getDistortedImage(self):
        """
        Get the distorted image from the simulator.
        """
        im = np.array(self.sim.getDistorted(),copy=False)
        if len(im.shape) > 2 and im.shape[-1] == 1:
            im.shape = im.shape[:2]
        self.image = im
        return  im
    def getActualImage(self):
        """
        Get the source image to file.
        """
        param = self.param
        try:
           return self.sim.getActualImage( reflines=param.get( "reflines" ) )
        except:
           im = np.array( self.sim.getActual() )
           if self.verbose: print( "actual image", type(im) )
           if im.shape[2] == 1 : im.shape = im.shape[:2]
           return im
    def getActual(self):
        """
        Write the source image to file.
        """
        param = self.param
        name = self.name
        fn = os.path.join(param.get("directory"),"actual-" + str(name) + ".png" ) 
        im = self.sim.getActualImage( reflines=param.get( "reflines" ) )
        cv.imwrite(fn,im)

    def getImage(self,centred=None,cropsize=None,reflines=None,verbose=None):
        if verbose is None: verbose = self.verbose
        if centred is None:
            centred = self.param.get( "centred" )
            if verbose: print( "[getImage] centring from parameters:", centred )
        if reflines is None:
            reflines = self.param.get( "reflines" )
        if centred:
            im = self.centreimage.copy()
        else:
            im = self.image.copy()
        if cropsize is None:
            cropsize = self.param.get( "cropsize" )
        if cropsize:
            im = crop(im,int( cropsize ), verbose=self.verbose  )
        if reflines:
            drawAxes(im)
        return im
    def saveImage(self):
        im = self.getImage()
        fn = self.param.get( ( "management", "filename" ), "test.png" )
        dir = self.param.get( ( "dataset", "directory" ) )
        if dir is not None:
            fn = os.path.join(dir, fn )
        if self.verbose: print( "[saveImage] saving to", fn)
        cv.imwrite(fn,im)
