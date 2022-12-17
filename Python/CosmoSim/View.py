#! /usr/bin/env python3
# (C) 2022: Hans Georg Schaathun <georg@schaathun.net> 

"""
The view (ttk Frame) for the desktop application.
"""

from tkinter import *
from tkinter import ttk
from PIL import Image, ImageTk
import numpy as np

class ImagePane(ttk.Frame):
    """
    A pane with all images to be displayed from the simulator.
    """
    def __init__(self,root,sim, *a, **kw ):
        """
        Set up the pane.

        :param root: parent widget
        :param sim: CosmoSim object
        """
        super().__init__(root, *a, **kw)
        self.sim = sim
        self.actual = Canvas(self,width=512,height=512)
        self.actual.grid(column=0,row=0)
        self.distorted = Canvas(self,width=512,height=512)
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
