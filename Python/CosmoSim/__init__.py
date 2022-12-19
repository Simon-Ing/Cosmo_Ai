# (C) 2022: Hans Georg Schaathun <georg@schaathun.net> 

import CosmoSim.CosmoSimPy as cs
import numpy as np
import threading as th

LensSpec = cs.LensSpec
SourceSpec = cs.SourceSpec

lensValues = {
        "Point Mass (exact)" : LensSpec.PointMass,
        "Point Mass (roulettes)" : LensSpec.PointMassRoulettes,
        "SIS (roulettes)" : LensSpec.SIS,
        }
sourceValues = {
        "Spherical" : SourceSpec.Sphere,
        "Ellipsoid" : SourceSpec.Ellipse,
        "Triangle" : SourceSpec.Triangle }


class CosmoSim(cs.CosmoSim):
    """
    Simulator for gravitational lensing.
    This wraps the CosmoSim library written in C++.  In particular,
    it wraps functions returning images, to convert the data to 
    numpy arrays.
    """
    def __init__(self,*a,**kw):
        super().__init__(*a,**kw)
        self._continue = True
        self.updateEvent = th.Event()
        self.simEvent = th.Event()
        self.simThread = th.Thread(target=self.simThread)
        self.simThread.start()
    def close(self):
        """
        Terminate the worker thread.
        """
        print ( "CosmoSim object closing" )
        self._continue = False
        self.simEvent.set()
        self.simThread.join()
        print ( "CosmoSim object closed" )
    def getUpdateEvent(self):
        return self.updateEvent
    def setSourceMode(self,s):
        return super().setSourceMode( int( sourceValues[s] ) ) 
    def setLensMode(self,s):
        return super().setLensMode( int( lensValues[s] ) ) 
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
        return im
    def getActualImage(self,reflines=True):
        """
        Return the Actual Image from the simulator as a numpy array.
        """
        im = np.array(self.getActual(reflines),copy=False)
        if im.shape[2] == 1 : im.shape = im.shape[:2]
        return im
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
        return im
