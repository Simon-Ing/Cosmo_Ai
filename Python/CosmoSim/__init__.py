# (C) 2022: Hans Georg Schaathun <georg@schaathun.net> 

import CosmoSim.CosmoSimPy as cs
import numpy as np

class CosmoSim(cs.CosmoSim):
    """
    Simulator for gravitational lensing.
    This wraps the CosmoSim library written in C++.  In particular,
    it wraps functions returning images, to convert the data to 
    numpy arrays.
    """
    def getActualImage(self):
        """
        Return the Actual Image from the simulator as a numpy array.
        """
        im = np.array(self.getActual(),copy=False)
        if im.shape[2] == 1 : im.shape = im.shape[:2]
        return im
    def getDistortedImage(self):
        """
        Return the Distorted Image from the simulator as a numpy array.
        """
        im = np.array(self.getDistorted(),copy=False)
        if im.shape[2] == 1 : im.shape = im.shape[:2]
        return im
