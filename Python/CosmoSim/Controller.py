#! /usr/bin/env python3
# (C) 2022: Hans Georg Schaathun <georg@schaathun.net> 

"""
Desktop application to run the CosmoSim simulator for
gravitational lensing.
"""

from tkinter import *
from tkinter import ttk
import math

from CosmoSim import lensValues, sourceValues

# Classes
class IntSlider:
    """
    A slider for integer values with label.
    """
    def __init__(self,root,text,row=1,fromval=0,toval=100,var=None,
            resolution=1, default=0):
        """
        Set up the slider with the following parameters:

        :param root: parent widget
        :param text: label text
        :param row: row number in the parent grid
        :param fromval: lower bound on the range
        :param toval: upper bound on the range
        :param var: variable object to use for the value 
                    (IntVar instanceby default)
        :param resolution: resolution of the variable (default 1)
        """
        if var == None:
            self.var = IntVar()
        else:
            self.var = var
        self.var.set( default )
        self.label = ttk.Label( root, text=text,
                style="Std.TLabel" )
        self.slider = Scale( root, length=250, variable=self.var,
                resolution=resolution,
                orient=HORIZONTAL,
                from_=fromval, to=toval )
        self.label.grid(row=row,column=0,sticky=E)
        self.slider.grid(row=row,column=1)
        # self.val = ttk.Label( root, textvariable=self.var )
        # self.val.grid(row=row,column=2)
    def get(self):
        "Get the value of the slider."
        return self.var.get()
    def set(self,v): 
        "Set the value of the slider."
        return self.var.get(v)
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
                values=[ "Spherical", "Ellipsoid", "Triangle" ] )
        sourceLabel.grid(column=0, row=1, sticky=E )
        self.sourceSelector.grid(column=1, row=1)

        self.sigmaSlider = IntSlider( self,
                text="Source Size", row=2,
                default=20 )
        self.sigma2Slider = IntSlider( self,
                text="Secondary Size", row=3,
                default=10 )
        self.thetaSlider = IntSlider( self, 
                toval=360,
                text="Source Rotation", row=4, default=45 )
        self.sigmaSlider.var.trace_add( "write", self.push)
        self.sigma2Slider.var.trace_add( "write", self.push)
        self.thetaSlider.var.trace_add( "write", self.push)
        self.push(runsim=False)
        modeVar.trace_add("write", self.push) 
    def push(self,*a,runsim=True):
        print( "[CosmoGUI] Push source parameters" )
        self.sim.setSourceParameters(
                self.sigmaSlider.get(),
                self.sigma2Slider.get(),
                self.thetaSlider.get()
                )
        self.sim.setSourceMode(self.sourceVar.get())
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
    def push(self,*a,runsim=True):
        print( "[CosmoGUI] Push image resolution" )
        self.sim.setImageSize( self.sizeSlider.get())
        self.sim.setResolution( self.resolutionSlider.get())
        self.sim.setBGColour( self.bgSlider.get())
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

        modeVar = StringVar()
        self.lensVar = modeVar
        modeVar.set( self.lensValues[0] )
        self.lensLabel = ttk.Label( self, text="Lens Model",
                style="Std.TLabel" )
        self.lensSelector = ttk.Combobox( self, 
                textvariable=modeVar,
                values=self.lensValues )
        self.lensLabel.grid(column=0, row=1, sticky=E )
        self.lensSelector.grid(column=1, row=1)
        self.lensSelector.set( self.lensValues[0] )

        self.einsteinSlider = IntSlider( self,
            text="Einstein Radius", row=2,
            default=20 )
        self.chiSlider = IntSlider( self,
            text="Distance Ratio (chi)", row=3,
            default=50 )
        self.ntermsSlider = IntSlider( self, 
            text="Number of Terms (Roulettes only)", row=4,
            toval=50,
            default=16 )
        self.einsteinSlider.var.trace_add( "write", self.push ) 
        self.chiSlider.var.trace_add( "write", self.push ) 
        self.ntermsSlider.var.trace_add( "write", self.push ) 

        self.maskModeVar = BooleanVar()
        self.maskModeVar.set( False )

        print ( "Ready to push parameters to Simulator" )

        self.push(runsim=False)
        print ( "Pushed parameters to Simulator" )
        self.maskModeVar.trace_add( "write",self.push )
        modeVar.trace_add("write", self.push) 
    def getMaskModeVar(self):
        return self.maskModeVar
    def push(self,*a,runsim=True):
        print( "[CosmoGUI] Push lens parameters" )
        self.sim.setLensMode(self.lensVar.get())
        self.sim.setNterms( self.ntermsSlider.get() )
        self.sim.setCHI( self.chiSlider.get() )
        self.sim.setEinsteinR( self.einsteinSlider.get())
        self.sim.setMaskMode( self.maskModeVar.get())
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

