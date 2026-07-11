# (C) 2026: Hans Georg Schaathun <georg@schaathun.net>

import cv2 as cv
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from ..Image import crop, annotatePoint, annotateCircle
from .. import Image as csimg
from .. import getMS

from .Generators import getSimulator, getLens, getSource
from .Parser import RouletteParser

from .Simulator import GenericSim
from .Arguments import Parameters

class SimImage(GenericSim):
    """
    This class simulates a single image.
    Once the simulation has been run, various kinds of image and metadata can be
    retrieved from the object.
    """
    def __init__(self,param,**kw):
        """
        In the present implementation, all the parameters must be passed as 
        a `Parameters` object.  The settings cannot be updated in an exising
        instance.
        """
        super().__init__(param,**kw)
        if self.verbose: print( f"[SimImage] init (verbose={self.verbose}) ..." )
        self.initSim()
        self.runSim()
    def getXiOffset(self,centrepoint):
        nu = self.sim.getNu()
        a = np.array(nu)
        return ( a[0] - centrepoint[0], a[1] - centrepoint[1] )

    def getData(self,precision=None,fn=None,verbose=None):
        """
        Get the data generated from the simulation, particularly the roulette
        amplitudes.  The return value is a pandas `Series` object.
        """
        if verbose is None: verbose = self.verbose
        sim = self.sim
        if self.param.get( "centred" ):
            if verbose: print( "[getData] centred" )
            centrepoint = self.centrepoint
        else:
            if verbose: print( "[getData] not centred" )
            centrepoint = (0,0)
        maxm = self.param.get( ( "simulator", "nterms" ), 16 )
        xireference = self.param.get( "xireference", True )

        releta = np.array(sim.getRelativeEta(centrepoint[0],centrepoint[1]))
        offset = np.array(sim.getOffset())
        if verbose: print( "[getData] ", offset, centrepoint )

        xioffset = self.getXiOffset(centrepoint)
        if verbose>1:
            print( f"[xireference={xireference}] nterms={maxm}, precision={precision}" )
        if precision is None:
            if xireference:
                ab = self.getAlphaBetas(maxm)
            else:
                ab = self.getAlphaBetas(maxm,pt=centrepoint)
        else:
            if xireference:
                xi = sim.getNu()
            else:
                xi = centrepoint
            if fn is None:
                fn = self.getAmplitudeFile() 
            g = self.param.get( ( "lens", "einsteinradius" ) )
            if verbose: print( "Einstein radius", g )
            rp = RouletteParser( fn, g=g, verbose=self.verbose )
            ab = rp.getAlphaBetas(xi,maxm=maxm)

        r1 = pd.Series(
                { "filename" : self.param.get( "filename" )
                , "source" : self.param.get( "source" )
                , "x" : self.param.get( "x" )
                , "y" : self.param.get( "y" )
                , "sigma" : self.param.get( "sigma" )
                , "sigma2" : self.param.get( "sigma2" )
                , "theta" : self.param.get( "theta" )
                , "lensX" : -centrepoint[0]
                , "lensY" : -centrepoint[1]
                , "centreX" : centrepoint[0]
                , "centreY" : centrepoint[1]
                , "reletaX" : releta[0]
                , "reletaY" : releta[1]
                , "offsetX" : offset[0]
                , "offsetY" : offset[1]
                , "xiX" : xioffset[0]
                , "xiY"  : xioffset[1]
                } )
        r1 = pd.concat( [ r1, ab ] )
        r1.name = self.param.get( "filename" )

        if verbose > 1:
            print( f"New row (verbosity={verbose})" )
            print( r1 )
        return r1
    def initSim(self):
        """
        Initialise the simulator, using the settings from the `param` attribute.
        """
        param = self.param
        self.src = getSource(self.param,verbose=self.verbose)
        self.lens = getLens(self.param,verbose=self.verbose)
        self.sim = getSimulator(self.param,lens=self.lens,source=self.src,
                                verbose=self.verbose)
        if param.get("y") is not None:
            if self.verbose > 1:
                print( "[initSim] XY", param.get( "x" ), param.get( "y" ) )
            self.sim.setXY( param.get( "x" ), param.get( "y" ) )
        elif param.get("phi",None) != None:
            if self.verbose > 1: 
                print( "[initSim] Polar", param.get( "x" ), param.get( "phi" ) )
            self.sim.setPolar( param.get( "x" ), param.get( "phi" ) )

    def moveImage(self,rot,scale,name):
        """
        Simulate with a different reference point.  This only makes
        sense for roulette models.
        """
        sim = self.sim
        param = self.param
        sim.moveSim(rot,scale)
        im = sim.getDistortedImage( 
                 critical=param.get( "criticalcurves" ),
                 showmask=param.get( "showmask" ) ) 
        print( "getDistortedImage() has returned", type(im) )

        if param.get( "centred" ):
            if self.verbose: print( "[movedImage] centred" )
            (im,(cx,cy)) = csimg.centreImage(im)
        if param.get( "cropsize" ):
            cs = param.get( "cropsize", None )
            if cs is not None:
                im = crop( int( cs ), verbose=self.verbose  )
        if param.get( "reflines" ):
            csimg.drawAxes(im)

        fn = os.path.join(param.get("directory"), str(name) + ".png" ) 
        cv.imwrite(fn,im)
        return im

    def getAmplitudeFile(self):
        fn = self.param.get( ( "lens", "amplitudefile" ) )
        if fn is None: 
            raise NotImplemented( "Not implemented lookup of default amplitudes file." )
        return fn
    def getAlphaBetas(self,maxm=2,pt=None):
        """
        Get the roulette amplitudes for a given point in the source plane.
        """
        if self.verbose>1:
            print ( "[getAlphaBetas] pt=", pt, "in Planar Co-ordinates"  )
        if pt == None:
           ab1 = { f"alpha[{m}][{s}]" : self.lens.getAlphaXi(m,s) 
                   for (m,s) in getMS(maxm) }
           ab2 = { f"beta[{m}][{s}]" : self.lens.getBetaXi(m,s)
                   for (m,s) in getMS(maxm) }
        else:
            (x,y) = pt
            # Scaling is done in getAlpha/getBeta
            ab1 = { f"alpha[{m}][{s}]" : self.lens.getAlphaPy(x,y,m,s)
                    for (m,s) in getMS(maxm) }
            ab2 = { f"beta[{m}][{s}]" : self.lens.getBetaPy(x,y,m,s)
                    for (m,s) in getMS(maxm) }
        return pd.concat( [ pd.Series( ab1 ), pd.Series( ab2 ) ] )

    def getAnnotated(self,
                     critical=(128,128,64),
                     centrePoint=(255,64,64),
                     xiOffset=(64,64,255),
                     convergenceRing=(64,64,255),
                     centred=None,cropsize=None):
        """
        Get an image with annotations showing key points and the convergence ring.
        This is incomplete and should be extended with addition annotations and
        options.
        """
        if centred is not None:
            raise NotImplementedError("centred option for getAnnotated() is not implemented yet.")
        im = self.getDistortedImage( )
        if critical is not None:
            crit = np.array( self.sim.getCritical() )
            crit = crit.astype( np.float64 )
            crit /= 255
            r,g,b = critical
            crit = np.dstack( [ crit*r, crit*g, crit*b ] )
            crit = crit.astype( np.uint8 )
            im = csimg.overlay( im, crit )
        
        if centrePoint is not None:
            im = annotatePoint( im, self.centrepoint, colour=centrePoint )
        pt = self.getXiOffset( (0,0) )
        x, y = pt
        if xiOffset is not None:
            im = annotatePoint( im, pt, colour=xiOffset )
        if convergenceRing is not None:
            convradius = np.sqrt( x*x + y*y )
            im = annotateCircle( im, pt, radius=convradius, colour=( 64, 64, 255 ) )
        if cropsize is None:
            cropsize = self.param.get( "cropsize" )
        if cropsize:
            im = crop(im,int( cropsize ), verbose=self.verbose  )
        return im
