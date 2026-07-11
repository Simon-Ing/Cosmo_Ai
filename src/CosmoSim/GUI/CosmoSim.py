# (C) 2022-26: Hans Georg Schaathun <georg@schaathun.net> 

"""
The CosmoSim package comprises several components, including scripts,
GUI, and API.

The root module provides wrappers around most of the C++ classes,
to make the python API more streamlined.
"""

from .. import CosmoSimPy as cs, getPathFN
from ..Sources import *
from ..Lens import *
from ..Dictionary import *
from ..CosmoSimPy import RouletteModel,RaytraceModel
from ..Image import drawAxes
import numpy as np
import threading as th
import os, sys

class CosmoSim:
    """
    Simulator for gravitational lensing.
    This wraps the CosmoSim library written in C++.  In particular,
    it wraps functions returning images, to convert the data to 
    numpy arrays.
    """
    def __init__(self,*a,maxm=50,verbose=1,**kw):
        super().__init__(*a,**kw)
        self.verbose = verbose
        if self.verbose>1: print( f"[CosmoSim] init (verbose={self.verbose}) ..." )

        # Default parameters
        self.amplitudefiles = { PsiSpec.PM : getPathFN( "pm50.txt" )
                 , PsiSpec.SIS : getPathFN( "sis50.txt" )
                 , PsiSpec.SIE : getPathFN( "sie05.txt" )
                 }
        self.bgcolour = 0
        self.imagesize = 512
        self.resolution = self.imagesize
        self._sampling = False

        self.simparam = { "nterms" : 5
             , "maskmode" : False
             } 
        self.lensparam = { "einsteinradius" : 30
             , "ellipseratio" : 0.6
             , "orientation" : 45
             } 
        self.srcparam = { "source" : "Spherical", "sigma" : 10 }

        # Instantiate simulator and objects
        self._sim = None
        self._lens = None
        self._psilens = None
        self._src = None
        self.setLensMode( PsiSpec.SIS )
        self.makeSource( )
        self.setModelMode( "Raytrace" )

        # Set up thread management
        self._continue = True
        self.updateEvent = th.Event()
        self.simEvent = th.Event()
        self._simThread = th.Thread(target=self.simThread)
        self._simThread.start()

    # Simulator
    def setModelMode(self,model):
        """
        Instantiate the simulator using the given model.
        It assumes that the lens and source have already been instantiated.
        """
        
        simmode = modelValues[model]
        if self.verbose:
            print( f"[setModelMode ({model})] instantiate new simulator")

        if simmode == ModelSpec.Raytrace:
            sim = RaytraceModel()
        elif simmode == ModelSpec.Roulette:
            sim = RouletteModel()
        else:
            raise RuntimeError( "Invalid simulator mode" )
        self._sim = sim
        self.setSimParameters()
        sim.setLens( self._lens )
        if self.verbose:
            print( f"[setModelMode ({model})] returns")
    def setSimParameters(self, param=None ):
        if param is None:
            param = self.simparam 
        else:
            self.simparam = param
        nterms = param.get( "nterms", None ) 
        if nterms is not None:
              self._sim.setNterms( nterms )
        mm = param.get( "maskmode", None ) 
        if mm is not None:
              self._sim.setMaskMode( mm )
        mr = param.get( "maskradius", None )
        if mr is not None:
              self._sim.setMaskRadius( mr )
    def setImageParameters(self, param=None ):
        self.resolution = param.get( "resolution", self.resolution )
        self.bgcolour = param.get( "bgcolour", self.bgcolour )
        self.imagesize = param.get( "imagesize", self.imagesize )

    # Lens
    def setLensMode(self,lensmode,*a,**kw):
        """
        Instantiate a new lens of the given mode.
        If a simulator is defined, the new lens is added thereto.
        """
        if self.verbose:
            print( f"[setLensMode({lensmode})] instantiate" )
        if lensmode == PsiSpec.PM:
            lens = PointMass()
        elif lensmode == PsiSpec.SIS:
            lens = SIS()
        elif lensmode == PsiSpec.SIE:
            lens = SIE()
        else:
            raise RuntimeError( "Invalid lens mode" )
        self._psilens_ = self._psilens # backup to prevent garbage collection
        self._psilens = lens
        lens.setFile( self.amplitudefiles[lensmode] )
        self.setLensParameters()
        if self._sim is not None:
            if self.verbose:
                print( f"[setLensMode({lensmode})] add lens to simulator" )
            self._sim.setLens( self._lens )
        if self.verbose:
            print( f"[setLensMode({lensmode})] returns" )
    def setLensParameters(self, param=None ):
        if param is None:
            param = self.lensparam 
            if self.verbose:
                print( "[setLensParameters]", param )
        else:
            self.lensparam = param
            if self.verbose:
                print( "[setLensParameters] new parameters" )
                print( param )
        einsteinradius = param.get( "einsteinradius", None )
        if einsteinradius is not None:
              self._psilens.setEinsteinR( einsteinradius )
        ratio = param.get( "ellipseratio", None ) 
        if ratio is not None:
              self._psilens.setRatio( ratio )
        orientation = param.get( "orientation", None ) 
        if orientation is not None:
              self._psilens.setOrientation( orientation )
        self.setSampled()
        if self.verbose:
            print( "[setLensParameters] returns" )
    def setSampled(self,sampling=None,**kw):
        """
        If `sampling` is given, the setting is updated.
        Then, if sampling is true, a sampled lens is created.
        The lens attribute is updated; if there is no sampled lens,
        it is set equal to `psilens`.
        """
        if sampling is None:
            sampling = self._sampling
        else:
            self._sampling = sampling
        _lens_ = self._lens  # Prevent garbage collection
        if sampling:
            size = self.imagesize
            if self.verbose>1: print( f"[initLens] Sampling (imagesize {size})" )
            self._lens = SampledPsiFunctionLens( self._psilens, size )
        else:
            self._lens = self._psilens

    # Source
    def makeSource(self,param=None):
        """
        Instantiate a new source according to the given parameters `param`.
        If a simulator is defined, the new source is added thereto.
        """
        if param is None:
            param = self.srcparam
        else:
            self.srcparam = param
        if param.get( "imagesize" ) == None:
           param.__setitem__( "imagesize", self.imagesize )
        _src_ = self._src # backup to prevent garbage collection
        self._src = getSource(param,verbose=self.verbose)

    # Source Position
    def setXY(self,x,y):
        """
        Set the source position using Cartesian Coordinates.
        """
        return self._sim.setXY(x,y)
    def setPolar(self,x,y): 
        """
        Set the source position using Polar Coordinates.
        """
        return self._sim.setPolar(x,y)


    def close(self):
        """
        Terminate the worker thread.
        This should be called before terminating the program,
        because stale threads would otherwise block.
        """
        self._continue = False
        self.simEvent.set()
        self._simThread.join()
    def getUpdateEvent(self):
        return self.updateEvent
    def maskImage(self,scale=1):
        return super().maskImage( float(scale) )

    def setBGColour(self,s):
        self.bgcolour = s
    def simThread(self):
        """
        This function repeatedly runs the simulator when the parameters
        have changed.  It is intended to run in a dedicated thread.
        """
        while self._continue:
            self.simEvent.wait()
            if self._continue:
               self.simEvent.clear()
               _src = self._src
               self._sim.setSource( _src )
               self._sim.update()
               self.updateEvent.set()
    def runSimulator(self):
        """
        Run the simulator; that is, tell it that the parameters
        have changed.  This triggers an event which will be handled
        when the simulator is idle.
        """
        if self.verbose: print( "CosmoSim.runSimulator() [python]" )
        self.simEvent.set()
        if self.verbose: print( "CosmoSim.runSimulator() triggered event" )

    def getActualImage(self,reflines=True,caustics=False):
        """
        Return the Actual Image from the simulator as a numpy array.
        """
        im = self._sim.getActual()
        if self.resolution < self.imagesize:
            im = cv.resize( im, ( self.resolution, self.resolution ) )
        if caustics:
            self._sim.drawCaustics( im )
        im = np.array(im)
        if reflines:
            csimg.drawAxes( im )
        if im.shape[2] == 1 :
            im.shape = im.shape[:2]
        return np.maximum(im,self.bgcolour)
    def showMask(self): return self._sim.markMask()
    def maskImage(self,scale): return self._sim.maskImage( scale )
    def getDistortedImage(self,reflines=False,critical=False,mask=False,showmask=False):
        """
        Return the Distorted Image from the simulator as a numpy array.
        """
        if self.verbose:
            print( "[getDistortedImage] Einstein Radius", self._psilens.getEinsteinR() )
        try:
            if mask: self.maskImage()
            if showmask: self.showMask()
        except:
            print( "Masking not supported for this lens model." )
        im = self._sim.getDistorted()
        if self.resolution < self.imagesize:
            im = cv.resize( im, ( self.resolution, self.resolution ) )
        if critical:
            self._sim.drawCritical( im )
        im = np.array(im)
        if reflines:
            drawAxes( im )
        if im.shape[2] == 1:
            im.shape = im.shape[:2]
        if im.shape[0]*im.shape[1] == 0:
            print( "[getDistortedImage] No image yet" )
        return np.maximum(im,self.bgcolour)

