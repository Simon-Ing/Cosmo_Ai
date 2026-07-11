#! /usr/bin/env python3
# (C) 2022: Hans Georg Schaathun <georg@schaathun.net> 

"""
Desktop application to run the CosmoSim simulator for
gravitational lensing.
"""

from tkinter import *
from tkinter import ttk
import math

from .Widgets import *
from ..Dictionary import *

class Controller(ttk.Frame):
    """
    Pane with widgets to control the various parameters for the simulation.
    """
    def getMaskModeVar(self): return self.lensFrame.getMaskModeVar()
    def __init__(self,root,sim, *a, **kw):
        """
        Set up the pane.

        :param root: parent widget
        :param sim: CosmoSim object
        """
        super().__init__(root, *a, **kw)
        self.sim = sim
        self.lensFrame = LensPane(self, sim, padding=10)
        self.lensFrame.grid(column=0,row=1)
        self.sourceFrame = SourcePane(self, sim, padding=10)
        self.sourceFrame.grid(column=1,row=1)
        self.posPane = PosPane(self,self.sim, padding=10)
        self.posPane.grid(column=2,row=1)

class SourcePane(ttk.Frame):
    """
    The pane of widgets to set the source parameters.
    """
    def __init__(self,root,sim, *a, **kw):
        """
        Set up the pane for the lens controls.

        :param root: parent widget
        :param sim: CosmoSim object
        """
        super().__init__(root, *a, **kw)
        self.sim = sim 
        self.sourceValues = list(sourceValues.keys())

        modeVar = StringVar()
        self.sourceVar = modeVar
        modeVar.set( self.sourceValues[0] )
        sourceLabel = ttk.Label( self,
            text="Source Model", style="Std.TLabel" )
        self.sourceSelector = ttk.Combobox( self,
                textvariable=modeVar,
                values=self.sourceValues ) # [ "Spherical", "Ellipsoid", "Triangle" ] )
        sourceLabel.grid(column=0, row=1, sticky=E )
        self.sourceSelector.grid(column=1, row=1)
        lightVar = StringVar()
        self.lightprfVar = lightVar
        lightVar.set( "Sersic" )
        lightProfileLabel = ttk.Label( self,
            text="Light Profile", style="Std.TLabel" )
        self.lightProfileSelector = ttk.Combobox( self,
                textvariable=lightVar,
                values=[ "Gaussian", "Sersic" ] )
        lightProfileLabel.grid(column=0, row=7, sticky=E )
        self.lightProfileSelector.grid(column=1, row=7)
        self.sigmaSlider = IntSlider( self,
                text="Source Size", row=2,
                default=20 )
        self.sigma2Slider = IntSlider( self,
                text="Secondary Size", row=3,
                default=10 )
        self.thetaSlider = IntSlider( self, 
                toval=360,
                text="Source Rotation", row=4, default=45 )
        self.nslider = FloatSlider( self,
                fromval=0.5,
                toval=10,
                text="Sersic Index", row=5, default=4 )
        self.Lslider = FloatSlider( self,
                toval=100,
                text= "Luminosity", row = 6,  default=10 )
        self.sigmaSlider.var.trace_add( "write", self.push)
        self.sigma2Slider.var.trace_add( "write", self.push)
        self.thetaSlider.var.trace_add( "write", self.push)
        self.nslider.var.trace_add("write", self.push)
        self.Lslider.var.trace_add("write", self.push)
        modeVar.trace_add("write", self.push) 
        lightVar.trace_add( "write", self.push)
        self.push(runsim=False)
    def push(self,*a,runsim=True):
        p = {
           "source": self.sourceVar.get(),
           "sigma":  self.sigmaSlider.get(),
           "sigma2":  self.sigma2Slider.get(),
           "theta":  self.thetaSlider.get(),
           "lightprofile": self.lightprfVar.get(),
           "n_sersic": self.nslider.get(),
           "luminosity": self.Lslider.get()
           }
        self.sim.makeSource(p)
        print( "[Controller.py] makeSource has returned" )
        if runsim: self.sim.runSimulator()

class ResolutionPane(ttk.Frame):
    """
    The pane of widgets to set the lens image resolution.
    """
    def __init__(self,root,sim, *a, **kw):
        """
        Set up the pane for the lens controls.

        :param root: parent widget
        :param sim: CosmoSim object
        """
        super().__init__(root, *a, **kw)
        self.sim = sim 

        self.sizeSlider = IntSlider( self, 
            text="Image Size", row=1,
            fromval=16,
            toval=1024,
            default=512 )
        self.sizeSlider.var.trace_add( "write", self.push )
        self.resolutionSlider = IntSlider( self, 
            text="Image Resolution", row=2,
            fromval=16,
            toval=1024,
            default=512 )
        self.resolutionSlider.var.trace_add( "write", self.push ) 
        self.bgSlider = IntSlider( self, 
            text="Background Colour", row=3,
            fromval=0,
            toval=255,
            default=3 )
        self.bgSlider.var.trace_add( "write", self.push ) 
        self.push(runsim=False)
    def push(self,*a,runsim=True):
        print( "[CosmoGUI] Push image resolution" )
        p = { "resolution" : self.resolutionSlider.get()
             , "bgcolour" : self.bgSlider.get()
             , "imagesize" : self.sizeSlider.get()
             } 
        print( p )
        self.sim.setImageParameters( p )
        if runsim: self.sim.runSimulator()

class LensPane(ttk.Frame):
    """
    The pane of widgets to set the lens parameters.
    """
    def __init__(self,root,sim, *a, **kw):
        """
        Set up the pane for the lens controls.

        :param root: parent widget
        :param sim: CosmoSim object
        """
        super().__init__(root, *a, **kw)
        self.sim = sim 

        self.lensValues = list(lensValues.keys())
        lensVar = StringVar()
        lensVar.set( self.lensValues[0] )
        self.lensVar = lensVar

        self.modelValues = list(modelValues.keys())
        simVar = StringVar()
        simVar.set( self.modelValues[0] )
        self.simVar = simVar

        self.sampleVar = BooleanVar()
        self.sampleVar.set( False )

        self.simLabel = ttk.Label( self, text="Simulator Model",
                style="Std.TLabel" )
        self.simSelector = ttk.Combobox( self, 
                textvariable=simVar,
                values=self.modelValues )

        self.lensLabel = ttk.Label( self, text="Lens Model",
                style="Std.TLabel" )
        self.lensSelector = ttk.Combobox( self, 
                textvariable=lensVar,
                values=self.lensValues )

        self.sampleButton = ttk.Checkbutton(self,
                onvalue=True, offvalue=False,
                variable=self.sampleVar,
                text="Sampled lens" )
        self.sampleButton.grid( column=1, row=2 )

        self.simLabel.grid(column=0, row=0, sticky=E )
        self.simSelector.grid(column=1, row=0)
        self.simSelector.set( self.modelValues[0] )
        self.lensLabel.grid(column=0, row=1, sticky=E )
        self.lensSelector.grid(column=1, row=1)
        self.lensSelector.set( self.lensValues[0] )

        self.einsteinSlider = IntSlider( self,
            text="Einstein Radius", row=3,
            default=20 )
        self.ratioSlider = FloatSlider( self,
            text="Ellipticity", row=4,
            fromval=0,
            toval=1,
            default=1 )
        self.orientationSlider = IntSlider( self, 
                toval=360,
                text="Lens Orientation", row=5, default=45 )
        self.ntermsSlider = IntSlider( self, 
            text="Number of Terms (Roulettes only)", row=7,
            toval=50,
            default=16 )
        self.einsteinSlider.var.trace_add( "write", self.pushLens ) 
        self.ratioSlider.var.trace_add( "write", self.pushLens ) 
        self.orientationSlider.var.trace_add( "write", self.pushLens ) 
        self.ntermsSlider.var.trace_add( "write", self.push ) 

        self.maskModeVar = BooleanVar()
        self.maskModeVar.set( False )

        self.push(runsim=False)
        self.pushLens(runsim=False)
        self.sim.setLensMode(lensValues[self.lensVar.get()])
        self.sim.setModelMode(self.simVar.get())
        self.sim.setSampled(self.sampleVar.get())
        print ( "Pushed parameters to Simulator" )

        self.maskModeVar.trace_add( "write",self.push )
        lensVar.trace_add("write", lambda *_ : 
            ( self.sim.setLensMode(lensValues[self.lensVar.get()])
            ,  self.sim.runSimulator() ) )
        simVar.trace_add("write", lambda *_ : 
            ( self.sim.setModelMode(self.simVar.get())
            ,  self.sim.runSimulator() ) )
        self.sampleVar.trace_add("write", lambda *_ : 
            ( self.sim.setSampled(self.sampleVar.get())
            ,  self.sim.runSimulator() )
            )
    def getMaskModeVar(self):
        return self.maskModeVar
    def pushLens(self,*a,runsim=True):
        print( "[CosmoGUI] Push lens parameters" )
        self.sim.setLensParameters(
            { "einsteinradius" : self.einsteinSlider.get()
             , "ellipseratio" : self.ratioSlider.get()
             , "orientation" : self.orientationSlider.get()
             } )
        if runsim: self.sim.runSimulator()
    def push(self,*a,runsim=True):
        print( "[CosmoGUI] Push lens parameters" )
        self.sim.setSimParameters(
            { "nterms" : self.ntermsSlider.get() 
             , "maskmode" : self.maskModeVar.get()
             } )
        if runsim: self.sim.runSimulator()
class PosPane(ttk.Frame):
    """
    The pane of widgets to set the source position.
    """
    def __init__(self,root,sim, *a, **kw):
        """
        Set up the pane.

        :param root: parent widget
        :param sim: CosmoSim object
        """
        super().__init__(root, *a, **kw)
        self.sim = sim 
        xSlider = IntSlider( self, text="x", row=1,
                fromval=-100,
                var=DoubleVar(), resolution=0.01 )
        ySlider = IntSlider( self, text="y", row=2,
                fromval=-100,
                var=DoubleVar(), resolution=0.01 )
        rSlider = IntSlider( self, text="r", row=3,
                var=DoubleVar(), resolution=0.01 )
        thetaSlider = IntSlider( self, text="theta", row=4,
                toval=360,
                var=DoubleVar(), resolution=0.1 )

        self.xVar = xSlider.var
        self.yVar = ySlider.var
        self.rVar = rSlider.var
        self.thetaVar = thetaSlider.var

        self.xVar.trace_add( "write", self.xyUpdate ) ;
        self.yVar.trace_add( "write", self.xyUpdate ) ;
        self.rVar.trace_add( "write", self.polarUpdate ) ;
        self.thetaVar.trace_add( "write", self.polarUpdate ) ;
        self._polarUpdate = False
        self._xyUpdate = False
        self.sim.setXY( self.xVar.get(), self.yVar.get() )
    def polarUpdate(self,*a):
        """
        Event handler to update Cartesian co-ordinates when polar 
        co-ordinates change.
        """
        self._polarUpdate = True
        if self._xyUpdate: 
            self._polarUpdate = False
            return
        r = self.rVar.get()
        theta = self.thetaVar.get()*math.pi/180
        self.xVar.set( math.cos(theta)*r ) 
        self.yVar.set( math.sin(theta)*r ) 
        self._polarUpdate = False
        self.push()
    def xyUpdate(self,*a):
        """
        Event handler to update polar co-ordinates when Cartesian 
        co-ordinates change.
        """
        self._xyUpdate = True
        if self._polarUpdate: 
            self._xyUpdate = False
            return
        x = self.xVar.get()
        y = self.yVar.get()
        r = math.sqrt( x*x + y*y ) 
        self.rVar.set( r )
        if r > 0:
           t = math.atan2( x, y )
           if t < 0: t += 2*math.pi
           self.thetaVar.set( t )
        self._xyUpdate = False
        self.push()
    def push(self):
        self.sim.setXY( self.xVar.get(), self.yVar.get() )
        self.sim.runSimulator()

