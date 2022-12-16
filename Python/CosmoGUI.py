#! /usr/bin/env python3

"""
Desktop application to run the CosmoSim simulator for
gravitational lensing.
"""

from tkinter import *
from tkinter import ttk
from PIL import Image, ImageTk
import numpy as np
import math

from CosmoSim import CosmoSim, lensValues, sourceValues

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
        self.slider = Scale( root, length=200, variable=self.var,
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

class Window:
    """
    The Window for the application
    """
    def mainloop(self): return self.root.mainloop()
    def destroy(self):
        self.sim.close()
        return self.root.destroy()
    def __init__(self,sim):
        self.root = Tk()
        self.sim = sim

        style = ttk.Style()
        style.configure("Red.TButton", foreground="white", background="red")
        labelstyle = ttk.Style()
        labelstyle.configure("Std.TLabel", foreground="black", padding=4,
                font=( "Arial", 15 ) )

        controller = Controller(self.root,sim)
        imgPane = ImagePane(self.root,sim)
        self.frm = ttk.Frame(self.root, padding=10)
        self.frm.grid()
        self.quitButton = ttk.Button(self.frm, text="Quit",
                command=self.destroy, style="Red.TButton")
        self.quitButton.grid(column=0, row=0, sticky=E)
class ImagePane:
    """
    A pane with all images to be displayed from the simulator.
    """
    def __init__(self,root,sim):
        """
        Set up the pane.

        :param root: parent widget
        :param sim: CosmoSim object
        """
        self.sim = sim
        self.frm = ttk.Frame(root, padding=10)
        self.frm.grid()
        self.actual = Canvas(self.frm,width=512,height=512)
        self.actual.grid(column=0,row=0)
        self.distorted = Canvas(self.frm,width=512,height=512)
        self.distorted.grid(column=1,row=0)
        im = Image.fromarray( np.zeros((512,512)) )
        img =  ImageTk.PhotoImage(image=im)
        self.actualCanvas = self.actual.create_image(0,0,anchor=NW, image=img)
        self.distortedCanvas = self.distorted.create_image(0,0,anchor=NW, image=img)
        sim.setCallback(self.update)
    def setActualImage(self):
        "Helper for `update()`."
        im0 = Image.fromarray( self.sim.getActualImage() )
        # Use an attribute to prevent garbage collection here
        self.img0 =  ImageTk.PhotoImage(image=im0)
        self.actual.itemconfig(self.actualCanvas, image=self.img0)
    def setDistortedImage(self):
        "Helper for `update()`."
        im1 = Image.fromarray( self.sim.getDistortedImage() )
        # Use an attribute to prevent garbage collection here
        self.img1 =  ImageTk.PhotoImage(image=im1)
        self.distorted.itemconfig(self.distortedCanvas, image=self.img1)
    def update(self):
        """
        Update the images with new data from the CosmoSim object.
        """
        self.setDistortedImage()
        self.setActualImage()
class Controller:
    """
    Pane with widgets to control the various parameters for the simulation.
    """
    def __init__(self,root,sim):
        """
        Set up the pane.

        :param root: parent widget
        :param sim: CosmoSim object
        """
        self.sim = sim
        self.lensValues = list(lensValues.keys())
        self.sourceValues = list(sourceValues.keys())
        self.frm = ttk.Frame(root, padding=10)
        self.frm.grid()
        self.lensFrame = ttk.Frame(self.frm, padding=10)
        self.lensFrame.grid(column=0,row=1)
        self.sourceFrame = ttk.Frame(self.frm, padding=10)
        self.sourceFrame.grid(column=1,row=1)
        self.posFrame = ttk.Frame(self.frm, padding=10)
        self.posFrame.grid(column=2,row=1)
        self.posPane = PosPane(self.posFrame,self.sim)
        self.makeLensFrame()
        self.makeSourceFrame()

    def makeLensFrame(self):
        modeVar = StringVar()
        self.lensVar = modeVar
        modeVar.set( self.lensValues[0] )
        self.lensLabel = ttk.Label( self.lensFrame, text="Lens Model",
                style="Std.TLabel" )
        self.lensSelector = ttk.Combobox( self.lensFrame, 
                textvariable=modeVar,
                values=self.lensValues )
        self.lensLabel.grid(column=0, row=1, sticky=E )
        self.lensSelector.grid(column=1, row=1)
        self.lensSelector.set( self.lensValues[0] )

        self.einsteinSlider = IntSlider( self.lensFrame,
            text="Einstein Radius", row=2,
            default=20 )
        self.chiSlider = IntSlider( self.lensFrame,
            text="Distance Ratio (chi)", row=3,
            default=50 )
        self.ntermsSlider = IntSlider( self.lensFrame, 
            text="Number of Terms (Roulettes only)", row=4,
            toval=50,
            default=16 )
        self.einsteinSlider.var.trace_add( "write", self.pushLensParameters ) 
        self.chiSlider.var.trace_add( "write", self.pushLensParameters ) 
        self.ntermsSlider.var.trace_add( "write", self.pushLensParameters ) 
        self.pushLensParameters(runsim=False)

        modeVar.trace_add("write", self.pushLensMode ) 

    def pushLensParameters(self,*a,runsim=True):
        print( "[CosmoGUI] Push lens parameters" )
        self.sim.setNterms( self.ntermsSlider.get() )
        self.sim.setCHI( self.chiSlider.get() )
        self.sim.setEinsteinR( self.einsteinSlider.get())
        if runsim: self.sim.runSimulator()
    def pushSourceParameters(self,*a,runsim=True):
        print( "[CosmoGUI] Push source parameters" )
        self.sim.setSourceParameters(
                self.sigmaSlider.get(),
                self.sigma2Slider.get(),
                self.thetaSlider.get()
                )
        if runsim: self.sim.runSimulator()
    def pushSourceMode(self,*a,runsim=True):
        self.sim.setSourceMode(self.sourceVar.get())
        if runsim: self.sim.runSimulator()
    def pushLensMode(self,*a,runsim=True):
        self.sim.setLensMode(self.lensVar.get())
        if runsim: self.sim.runSimulator()
    def makeSourceFrame(self):
        modeVar = StringVar()
        self.sourceVar = modeVar
        modeVar.set( self.sourceValues[0] )
        sourceLabel = ttk.Label( self.sourceFrame,
            text="Source Model", style="Std.TLabel" )
        self.sourceSelector = ttk.Combobox( self.sourceFrame,
                textvariable=modeVar,
                values=[ "Spherical", "Ellipsoid", "Triangle" ] )
        sourceLabel.grid(column=0, row=1, sticky=E )
        self.sourceSelector.grid(column=1, row=1)

        self.sigmaSlider = IntSlider( self.sourceFrame,
                text="Source Size", row=2,
                default=20 )
        self.sigma2Slider = IntSlider( self.sourceFrame,
                text="Secondary Size", row=3,
                default=10 )
        self.thetaSlider = IntSlider( self.sourceFrame, 
                toval=360,
                text="Source Rotation", row=4, default=45 )
        self.sigmaSlider.var.trace_add( "write", self.pushSourceParameters )
        self.sigma2Slider.var.trace_add( "write", self.pushSourceParameters )
        self.thetaSlider.var.trace_add( "write", self.pushSourceParameters )
        self.pushSourceParameters(runsim=False)
        modeVar.trace_add("write", self.pushSourceMode ) 

class PosPane:
    """
    The pane of widgets to set the source position.
    """
    def __init__(self,frm,sim):
        self.frm = frm 
        self.sim = sim 
        xSlider = IntSlider( self.frm, text="x", row=1,
                fromval=-100,
                var=DoubleVar(), resolution=0.01 )
        ySlider = IntSlider( self.frm, text="y", row=2,
                fromval=-100,
                var=DoubleVar(), resolution=0.01 )
        rSlider = IntSlider( self.frm, text="r", row=3,
                var=DoubleVar(), resolution=0.01 )
        thetaSlider = IntSlider( self.frm, text="theta", row=4,
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
        print( "polarUpdate", *a )
        r = self.rVar.get()
        theta = self.thetaVar.get()*math.pi/180
        self.xVar.set( math.cos(theta)*r ) 
        self.yVar.set( math.sin(theta)*r ) 
        self._polarUpdate = False
        self.pushXY()
    def xyUpdate(self,*a):
        """
        Event handler to update polar co-ordinates when Cartesian 
        co-ordinates change.
        """
        self._xyUpdate = True
        if self._polarUpdate: 
            self._xyUpdate = False
            return
        print( "xyUpdate", *a )
        x = self.xVar.get()
        y = self.yVar.get()
        r = math.sqrt( x*x + y*y ) 
        self.rVar.set( r )
        if r > 0:
           t = math.atan2( x, y )
           if t < 0: t += 2*math.pi
           self.thetaVar.set( t )
        self._xyUpdate = False
        self.pushXY()
    def pushXY(self):
        self.sim.setXY( self.xVar.get(), self.yVar.get() )
        self.sim.runSimulator()

if __name__ == "__main__":
    print( "CosmoGUI starting." )

    sim = CosmoSim()
    win = Window(sim)

    sim.runSimulator()

    win.mainloop()

