# (C) 2022: Hans Georg Schaathun <georg@schaathun.net> 

import CosmoSim.CosmoSimPy as cs
import numpy as np

class CosmoSim(cs.CosmoSim):
    def getActualImage(self):
        return np.array(self.getActual(),copy=False)
    def getDistortedImage(self):
        return np.array(self.getDistorted(),copy=False)
