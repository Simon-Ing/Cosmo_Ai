#! /usr/bin/env python3

"""
Desktop application to run the CosmoSim simulator for
gravitational lensing.

(Under construction.)
"""

from tkinter import *
from tkinter import ttk
from PIL import Image, ImageTk
import numpy as np
import math

from CosmoSim import CosmoSim


# Classes
class IntSlider:
    """
    A slider for integer values with label.
    """
    def __init__(self,root,text,row=1,fromval=0,toval=100):
        """
        Set up the slider with the following parameters:

        :param root: parent widget
        :param text: label text
        :param row: row number in the parent grid
        :param fromval: lower bound on the range
        :param toval: upper bound on the range
        """
        self.var = IntVar()
        self.label = ttk.Label( root, text=text,
                style="Std.TLabel" )
        self.slider = Scale( root, length=200, variable=self.var,
                resolution=1,
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
        "Update the images with new data from the CosmoSim object."
        self.setDistortedImage()
        self.setActualImage()
class Controller:
    lensValues = [ "SIS (roulettes)", "Point Mass (exact)",
                        "Point Mass (roulettes)" ]
    def __init__(self,root,sim):
        """
        Set up the pane.

        :param root: parent widget
        :param sim: CosmoSim object
        """
        self.sim = sim
        self.frm = ttk.Frame(root, padding=10)
        self.frm.grid()
        self.lensFrame = ttk.Frame(self.frm, padding=10)
        self.lensFrame.grid(column=0,row=1)
        self.sourceFrame = ttk.Frame(self.frm, padding=10)
        self.sourceFrame.grid(column=1,row=1)
        self.posFrame = ttk.Frame(self.frm, padding=10)
        self.posFrame.grid(column=2,row=1)
        self.quitButton = ttk.Button(self.frm, text="Quit",
                command=root.destroy, style="Red.TButton")
        self.quitButton.grid(column=2, row=0)
        self.makeLensFrame()
        self.makeSourceFrame()
        self.posPane = PosPane(self.posFrame,self.sim)

    def makeLensFrame(self):
        self.lensLabel = ttk.Label( self.lensFrame, text="Lens Model",
                style="Std.TLabel" )
        self.lensSelector = ttk.Combobox( self.lensFrame, 
                values=self.lensValues )
        self.lensLabel.grid(column=0, row=1, sticky=E )
        self.lensSelector.grid(column=1, row=1)

        self.einsteinSlider = IntSlider( self.lensFrame,
            text="Einstein Radius", row=2 )
        self.chiSlider = IntSlider( self.lensFrame,
            text="Distance Ratio (chi)", row=3 )
        self.nTermsSlider = IntSlider( self.lensFrame, 
            text="Number of Terms (Roulettes only)", row=4)

    def pushSourceParameters(self,*a):
        print( "[CosmoGUI] Push source parameters" )
        self.sim.setSourceParameters(
                0, # self.sourceSelector.var.get
                self.sigmaSlider.var.get(),
                self.sigma2Slider.var.get(),
                self.thetaSlider.var.get()
                )
        self.sim.runSimulator()
    def makeSourceFrame(self):
        sourceLabel = ttk.Label( self.sourceFrame,
            text="Source Model", style="Std.TLabel" )
        self.sourceSelector = ttk.Combobox( self.sourceFrame,
            values=[ "Spherical", "Ellipsoid", "Triangle" ] )
        sourceLabel.grid(column=0, row=1, sticky=E )
        self.sourceSelector.grid(column=1, row=1)

        self.sigmaSlider = IntSlider( self.sourceFrame,
                text="Source Size", row=2  )
        self.sigma2Slider = IntSlider( self.sourceFrame,
                text="Secondary Size", row=3 )
        self.thetaSlider = IntSlider( self.sourceFrame, 
                text="Source Rotation", row=4 )
        self.sigmaSlider.var.trace_add( "write", self.pushSourceParameters )
        self.sigma2Slider.var.trace_add( "write", self.pushSourceParameters )
        self.thetaSlider.var.trace_add( "write", self.pushSourceParameters )
    def updateSim(self):
        """
        Push parameters to the CosmoSim object
        """
        sim = self.sim
        x,y = self.posPane.getXY()
        sim.setXY( x, y )
        sim.setNterms( self.ntermsSlider.get() )
        sim.setCHI( self.chiSlider.get() )
        sim.setEinsteinR( self.einsteinSlider.get())
        # sim.setLensMode( self.einsteinSlider.get() )
        # sim.setSourceParameters( self.einsteinSlider.get() )
        # TODO: Complete

class PosPane:
    """
    The pane of widgets to set the source position.
    """
    def __init__(self,frm,sim):
        self.frm = frm 
        self.sim = sim 
        xSlider = IntSlider( self.frm, text="x", row=1 )
        ySlider = IntSlider( self.frm, text="y", row=2 )
        rSlider = IntSlider( self.frm, text="r", row=3 )
        thetaSlider = IntSlider( self.frm, text="theta", row=4 )

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
        self.sim.setXY( self.xVar.get(), self.yVar.get() )
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
        self.sim.setXY( self.xVar.get(), self.yVar.get() )

# Main object
root = Tk()

# Styles
style = ttk.Style()
style.configure("Red.TButton", foreground="white", background="red")
labelstyle = ttk.Style()
labelstyle.configure("Std.TLabel", foreground="black", padding=4,
        font=( "Arial", 15 ) )


# Simulator
sim = CosmoSim()
sim.runSim()

# GUI
controller = Controller(root,sim)
imgPane = ImagePane(root,sim)
imgPane.update()

# Main Loop
root.mainloop()

