# (C) 2022: Hans Georg Schaathun <georg@schaathun.net> 

import CosmoSim.CosmoSimPy as cs
import numpy as np
import threading as th
import os

LensSpec = cs.LensSpec
SourceSpec = cs.SourceSpec

lensDict = {
        "Point Mass (exact)" : LensSpec.PointMass,
        "Point Mass (roulettes)" : LensSpec.PointMassRoulettes,
        "SIS (rotated)" : LensSpec.SIS,
        "SIS (roulettes)" : LensSpec.SISRoulettes,
        "Sampled SIS" : LensSpec.SampledSIS,
        "PureSampled SIS" : LensSpec.PureSampledSIS,
        "Roulette SIS" : LensSpec.RouletteSIS,
        "PsiFunctionSIS" : LensSpec.PsiFunctionSIS,
        "p" : LensSpec.PointMass,
        "p" : LensSpec.PointMass,
        "r" : LensSpec.PointMassRoulettes,
        "s" : LensSpec.SIS,
        "sr" : LensSpec.SISRoulettes,
        "ss" : LensSpec.SampledSIS,
        "pss" : LensSpec.PureSampledSIS,
        "fs" : LensSpec.PsiFunctionSIS,
        "rs" : LensSpec.RouletteSIS,
        }
lensValues = {
        "Point Mass (exact)" : LensSpec.PointMass,
        "Point Mass (roulettes)" : LensSpec.PointMassRoulettes,
        "SIS (rotated)" : LensSpec.SIS,
        "SIS (roulettes)" : LensSpec.SISRoulettes,
        "Sampled SIS" : LensSpec.SampledSIS,
        "PureSampled SIS" : LensSpec.PureSampledSIS,
        "PsiFunctionSIS" : LensSpec.PsiFunctionSIS,
        "Roulette SIS (new)" : LensSpec.RouletteSIS,
        }
sourceDict = {
        "Spherical" : SourceSpec.Sphere,
        "Ellipsoid" : SourceSpec.Ellipse,
        "Triangle" : SourceSpec.Triangle,
        "s" : SourceSpec.Sphere,
        "e" : SourceSpec.Ellipse,
        "t" : SourceSpec.Triangle,
        }
sourceValues = {
        "Spherical" : SourceSpec.Sphere,
        "Ellipsoid" : SourceSpec.Ellipse,
        "Triangle" : SourceSpec.Triangle,
        }

def getMS(maxm): return [ (m,s) for m in range(maxm+1)
                                    for s in range(1-m%2,m+2,2) ]
def getMSheaders(maxm): 
    r = [ ( f"alpha[{m},{s}]", f"beta[{m},{s}]") for (m,s) in getMS(maxm) ]
    return [ x for p in r for x in p ]

maxmlist = [ 50, 100, 200 ]
def getFileName(maxm):
    """
    Get the filename for the amplitudes files.
    The argument `maxm` is the maximum number of terms (nterms) to be
    used in the simulator.
    """
    dir = os.path.dirname(os.path.abspath(__file__))
    for m in maxmlist:
        m0 = m
        if maxm <= m:
            return( os.path.join( dir, f"{m}.txt" ) )
    raise Exception( f"Cannot support m > {m0}." )
    

class CosmoSim(cs.CosmoSim):
    """
    Simulator for gravitational lensing.
    This wraps the CosmoSim library written in C++.  In particular,
    it wraps functions returning images, to convert the data to 
    numpy arrays.
    """
    def __init__(self,*a,maxm=50,fn=None,**kw):
        super().__init__(*a,**kw)
        if fn == None:
            super().setFile( getFileName( maxm ) )
        else:
            super().setFile( fn )
        self._continue = True
        self.updateEvent = th.Event()
        self.simEvent = th.Event()
        self.simThread = th.Thread(target=self.simThread)
        self.simThread.start()
        self.bgcolour = 0
    def getAlphas(self,maxm=2):
        return [ self.getAlphaXi(m,s) for (m,s) in getMS(maxm) ]
    def getBetas(self,maxm=2):
        return [ self.getBetaXi(m,s) for (m,s) in getMS(maxm) ]
    def getAlphaBetas(self,maxm=2,pt=None):
        if pt == None:
           r = [ (self.getAlphaXi(m,s),self.getBetaXi(m,s)) for (m,s) in getMS(maxm) ]
        else:
            (x,y) = pt
            r = [ (self.getAlpha(x,y,m,s),self.getBeta(x,y,m,s)) 
                    for (m,s) in getMS(maxm) ]
        return [ x for p in r for x in p ]

    def close(self):
        """
        Terminate the worker thread.
        This should be called before terminating the program,
        because stale threads would otherwise block.
        """
        print ( "CosmoSim object closing" )
        self._continue = False
        self.simEvent.set()
        self.simThread.join()
        print ( "CosmoSim object closed" )
    def getUpdateEvent(self):
        return self.updateEvent
    def setSourceMode(self,s):
        return super().setSourceMode( int( sourceDict[s] ) ) 
    def moveSim(self,rot,scale):
        return super().moveSim( float(rot), float(scale) )
    def maskImage(self,scale=1):
        return super().maskImage( float(scale) )
    def setLensMode(self,s):
        return super().setLensMode( int( lensDict[s] ) ) 
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
               self.runSim()
               self.updateEvent.set()
        print( "simThread() returning" )
    def runSimulator(self):
        """
        Run the simulator; that is, tell it that the parameters
        have changed.  This triggers an event which will be handled
        when the simulator is idle.
        """
        self.simEvent.set()

    def getApparentImage(self,reflines=True):
        """
        Return the Apparent Image from the simulator as a numpy array.
        """
        im = np.array(self.getApparent(reflines),copy=False)
        if im.shape[2] == 1 : im.shape = im.shape[:2]
        return np.maximum(im,self.bgcolour)
    def getActualImage(self,reflines=True):
        """
        Return the Actual Image from the simulator as a numpy array.
        """
        im = np.array(self.getActual(reflines),copy=False)
        if im.shape[2] == 1 : im.shape = im.shape[:2]
        return np.maximum(im,self.bgcolour)
    def getPsiMap(self):
        """
        Return a matrix representation of the sampled lensing potential.
        """
        im = np.array(super().getPsiMap(),copy=False)
        print(im.shape,im.dtype)
        if im.shape[2] == 1 : im.shape = im.shape[:2]
        return im
    def getMassMap(self):
        """
        Return a matrix representation of the sampled mass density.
        """
        # im = super().getMassMap()
        # print(type(im))
        im = np.array(super().getMassMap(),copy=False)
        print(im.shape,im.dtype)
        if im.shape[2] == 1 : im.shape = im.shape[:2]
        return im[2:-2,2:-2]
    def getDistortedImage(self,reflines=True,mask=False,showmask=False):
        """
        Return the Distorted Image from the simulator as a numpy array.
        """
        try:
            if mask: self.maskImage()
            if showmask: self.showMask()
        except:
            print( "Masking not supported for this lens model." )
        im = np.array(self.getDistorted(reflines),copy=False)
        if im.shape[2] == 1 : im.shape = im.shape[:2]
        return np.maximum(im,self.bgcolour)
