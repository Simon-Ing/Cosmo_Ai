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

from CosmoSim import CosmoSim


# Classes
class RoundedVar(DoubleVar):
    def set(self,v):
        vv = round(v)
        print(vv)
        return super().set(vv)

class IntSlider:
    def __init__(self,parent,text,row=1,fromval=0,toval=100):
        self.var = IntVar()
        self.label = ttk.Label( parent, text=text,
                style="Std.TLabel" )
        self.slider = Scale( parent, length=200, variable=self.var,
                resolution=1,
                orient=HORIZONTAL,
                from_=fromval, to=toval )
        # self.val = ttk.Label( parent, textvariable=self.var )
        self.label.grid(row=row,column=0,sticky=E)
        self.slider.grid(row=row,column=1)
        # self.val.grid(row=row,column=2)
    def get(self): return self.var.get()
    def set(self,v): return self.var.get(v)

class ImagePane:
    def __init__(self,root,sim):
        self.sim = sim
        self.actual = Canvas(root,width=512,height=512)
        self.actual.grid(column=0,row=0)
        self.distorted = Canvas(root,width=512,height=512)
        self.distorted.grid(column=1,row=0)
        im = Image.fromarray( np.zeros((512,512)) )
        img =  ImageTk.PhotoImage(image=im)
        self.actualCanvas = self.actual.create_image(0,0,anchor=NW, image=img)
        self.distortedCanvas = self.distorted.create_image(0,0,anchor=NW, image=img)
    def setActualImage(self):
        im0 = Image.fromarray( self.sim.getActualImage() )
        # Use an attribute to prevent garbage collection here
        self.img0 =  ImageTk.PhotoImage(image=im0)
        self.actual.itemconfig(self.actualCanvas, image=self.img0)
    def setDistortedImage(self):
        im1 = Image.fromarray( self.sim.getDistortedImage() )
        self.img1 =  ImageTk.PhotoImage(image=im1)
        self.distorted.itemconfig(self.distortedCanvas, image=self.img1)
    def update(self):
        self.setDistortedImage()
        self.setActualImage()

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
sim.init()
sim.runSim()

# GUI
frm1 = ttk.Frame(root, padding=10)
frm2 = ttk.Frame(root, padding=10)
frm1.grid()
frm2.grid()
lensFrame = ttk.Frame(frm1, padding=10)
lensFrame.grid(column=0,row=1)
sourceFrame = ttk.Frame(frm1, padding=10)
sourceFrame.grid(column=1,row=1)
posFrame = ttk.Frame(frm1, padding=10)
posFrame.grid(column=2,row=1)


quitButton = ttk.Button(frm1, text="Quit", command=root.destroy, style="Red.TButton")
quitButton.grid(column=2, row=0)

lensLabel = ttk.Label( lensFrame, text="Lens Model", style="Std.TLabel" )
lensSelector = ttk.Combobox( lensFrame, values=[ "SIS (roulettes)", "Point Mass (exact)", "Point Mass (roulettes)" ] )
lensLabel.grid(column=0, row=1, sticky=E )
lensSelector.grid(column=1, row=1)

einsteinSlider = IntSlider( lensFrame, text="Einstein Radius", row=2 )
chiSlider = IntSlider( lensFrame, text="Distance Ratio (chi)", row=3 )
nTermsSlider = IntSlider( lensFrame, text="Number of Terms (Roulettes only)",
        row=4)

sourceLabel = ttk.Label( sourceFrame, text="Source Model", style="Std.TLabel" )
sourceSelector = ttk.Combobox( sourceFrame, values=[ "Spherical", "Ellipsoid", "Triangle" ] )
sourceLabel.grid(column=0, row=1, sticky=E )
sourceSelector.grid(column=1, row=1)

sigmaSlider = IntSlider( sourceFrame, text="Source Size", row=2  )
sigma2Slider = IntSlider( sourceFrame, text="Secondary Size", row=3 )
thetaSlider = IntSlider( sourceFrame, text="Source Rotation", row=4 )

xSlider = IntSlider( posFrame, text="x", row=1 )
ySlider = IntSlider( posFrame, text="y", row=2 )






imgPane = ImagePane(frm2,sim)
imgPane.update()

root.mainloop()

