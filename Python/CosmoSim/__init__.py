# (C) 2022: Hans Georg Schaathun <georg@schaathun.net> 

import CosmoSim.CosmoSimPy as cs
import numpy as np

class CosmoSim(cs.CosmoSim):
    """
    Simulator for gravitational lensing.
    This wraps the CosmoSim library written in C++.
    """
    def getActualImage(self):
        im = np.array(self.getActual(),copy=False)
        if im.shape[2] == 1 : im.shape = im.shape[:2]
        return im
    def getDistortedImage(self):
        im = np.array(self.getDistorted(),copy=False)
        if im.shape[2] == 1 : im.shape = im.shape[:2]
        return im
