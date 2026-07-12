# (C) 2026: Hans Georg Schaathun <georg@schaathun.net>

import cv2 as cv
import numpy as np
import pandas as pd

from .Arguments import Parameters
from .Simulator import GenericSim 

from ..RouletteAmplitudes import RouletteAmplitudes 
from .Generators import RouletteRegenerator, getSource

class Resim(GenericSim):
    """
    Infrastructure to resimulate from roulette amplitudes.

    The class sets up the infrastructure, and provides the methods to
    run the simulator for each iaage on a CSV file.
    """
    def __init__(self,row,param=None,**kw):
        """
        Note that `args` overrides Boolean parameters.
        """
        if param is None:
            param = Parameters()
        super().__init__(param,**kw)
        if self.verbose>2: print( "[Resim.__init__]" )

        self.sim = RouletteRegenerator(verbose=self.verbose)
        self.xireference = self.param.get( "xireference", True )
        self.nterms = self.param.get( "nterms" )
        if self.verbose>1: print( "[Resim] nterms =", self.nterms )

        self.loadData( row )
        self.initSim()
        self.runSim()

        im = self.image
        if self.xireference:
              R = np.float32( [ [ 1, 0, row["xiX"] ], [ 0, 1, -row["xiY"] ] ] )
              m,n = im.shape
              try:
                  self.image = cv.warpAffine(im,R,(n,m))
              except Exception as e:
                  print( "Error in warpAffine.  Image", im )
                  print( "Image shape", (m,n) )
                  raise e
    def initSim(self):
        """
        Initialise the simulator with the given data row.
        """
        param = self.param
        row = self.row

        fn = param.get( "filename" )
        if fn is None: fn = row.name
        if self.verbose > 1: print( "[initSim] item ", fn)

        if self.verbose>2: print( "[initSim]" )
        maxm = self.coefs.getNterms()
        
        if self.xireference:
            if self.verbose:
                print( "xi", row["xiX"], row["xiY"], row["sigma"] )
            pt = (0,0)
        else:
            try:
                pt = ( row["offsetX"], row["offsetY"] )
            except:
                raise RuntimeError( "offsetX/offsetY missing from dataset [no-xireference-mode]" )
            if self.verbose:
                print( "Offset", row["offsetX"], row["offsetY"], row["sigma"] )
                print( row )
        self.sim.setCentrePy( *pt )
        if self.verbose: print( "Initialised simulator at point", pt )

        for m in range(maxm+1):
            for s in range((m+1)%2, m+2, 2):
                if self.verbose>1: print( "row is",  type( row ) )
                alpha = row.get( f"alpha[{m}][{s}]", 0.0 )
                beta = row.get( f"beta[{m}][{s}]", 0.0 )
                if self.verbose>1:
                    print( "alpha/beta are",  type( alpha ), type( beta ) )
                    print( f"alpha[{m}][{s}] = {alpha}\t\tbeta[{m}][{s}] = {beta}." )
                self.sim.setAlphaXi( m, s, alpha )
                self.sim.setBetaXi( m, s, beta )
        if self.verbose>1:
            print( "Source spec:", self.param.get( "source" ) )
        self.src = getSource( param )
        self.sim.setSource( self.src )

    def loadData( self, row ):
        """
        Load data as a row from the dataset.

        The `Parameters` class does not support roulette amplitudes, and hence
        they are handled separately.
        """
        if self.verbose>2: print( "[loadData]" )
        if not isinstance(row,pd.Series):
            raise RuntimeException( "Should have a pandas Series as a row object." )
        cols = row.axes[0]
        if self.verbose>1: print( "columns:", cols )
    
        coefs = RouletteAmplitudes(cols)

        nterms = self.param.get( ( "simulator", "nterms" ) )
        if nterms is None:
            if self.verbose: print( "[loadData] deducing nterms =", nterms )
            nterms = coefs.getNterms()
        elif self.verbose: print( "[loadData] nterms =", nterms )

        self.sim.setNterms( int(nterms) )

        self.param.setRow( row )
        self.coefs = coefs
        self.cols = cols
        self.row = row
