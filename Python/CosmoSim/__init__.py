# (C) 2022: Hans Georg Schaathun <georg@schaathun.net> 

import CosmoSim.CosmoSimPy as cs
import numpy as np

class CosmoSim(cs.CosmoSim):
    """
    Simulator for gravitational lensing.
    This wraps the CosmoSim library written in C++.
    """
    def getActualImage(self):
        return np.array(self.getActual(),copy=False)
    def getDistortedImage(self):
        return np.array(self.getDistorted(),copy=False)
