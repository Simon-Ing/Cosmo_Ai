#! /usr/bin/env python3
# (C) 2022: Hans Georg Schaathun <georg@schaathun.net> 

"""
The view (ttk Frame) for the desktop application.
"""

from tkinter import *
from tkinter import ttk
from PIL import Image, ImageTk
import numpy as np
import threading as th
from ..Image import drawAxes

import sys


class ImageCanvas(Canvas):
    def __init__(self,parent,image=None,verbose=1,**kwargs):
        super().__init__(parent,**kwargs)
        self.verbose = verbose
        self.height = self.winfo_reqheight()
        self.width = self.winfo_reqwidth()
        if None != image:
            im = image
        else:
            im = Image.fromarray( np.zeros((512,512)) )
        self.origimage = im
        self.img = ImageTk.PhotoImage(image=im)
        self.imageCanvas = self.create_image(0,0,anchor=NW, 
                image=self.img)
        self.bind("<Configure>", self.on_resize)

    def setImage(self,im=None):
        # Use an attribute to prevent garbage collection here
        if None != im:
           self.origimage = im
        else:
           im = self.origimage 
        size = self.height 
        self.im = im.resize((size,size), Image.NEAREST)
        if self.verbose > 1:
            print( "Image resized", size )
        self.img =  ImageTk.PhotoImage(image=self.im)
        self.itemconfig(self.imageCanvas, image=self.img)

    def on_resize(self,event):
        self.height = event.height
        self.setImage()

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
        self._continue = True
        self.actual = ImageCanvas(self,width=512,height=512)
        self.actual.grid(column=0,row=0)
        self.distorted = ImageCanvas(self,width=512,height=512)
        self.distorted.grid(column=1,row=0)
        self.height = 540

        self.criticalVar = BooleanVar()
        self.criticalVar.set( False )
        self.reflinesVar = BooleanVar()
        self.reflinesVar.set( True )
        self.maskVar = BooleanVar()
        self.maskVar.set( False )
        self.showmaskVar = BooleanVar()
        self.showmaskVar.set( False )

        self.updateEvent = sim.getUpdateEvent()
        self.updateThread = th.Thread(target=self.updateThread)
        self.updateThread.start()
        self.criticalVar.trace_add( "write", 
                lambda *a : self.updateEvent.set() )
        self.reflinesVar.trace_add( "write", 
                lambda *a : self.updateEvent.set() )
        self.maskVar.trace_add( "write",
                lambda *a : self.updateEvent.set() )
        self.showmaskVar.trace_add( "write",
                lambda *a : self.updateEvent.set() )
        self.bind("<Configure>", self.on_resize)
        self.updateEvent.set() 
    def on_resize(self,event):
        if np.abs(self.height - event.height) > 4:
           self.height = event.height
           size = self.height - 25
           self.actual.config(width=size, height=size)
           self.distorted.config(width=size, height=size)
    def getCriticalVar(self):
        return self.criticalVar
    def getReflinesVar(self):
        return self.reflinesVar
    def getMaskVar(self):
        return self.maskVar
    def getShowmaskVar(self):
        return self.showmaskVar
    def close(self):
        """
        Terminate the update thread.
        This should be called before terminating the program,
        because stale threads would otherwise block.
        """
        self._continue = False
        self.updateEvent.set()
        self.updateThread.join()
    def setActualImage(self):
        "Helper for `update()`."
        sys.stdout.flush()
        im = self.sim.getActualImage(reflines=False,caustics=self.criticalVar.get()) 
        if self.reflinesVar.get(): drawAxes(im)
        im0 = Image.fromarray(im)
        self.actual.setImage(im0)
    def setDistortedImage(self):
        "Helper for `update()`."
        im = self.sim.getDistortedImage( 
                critical=self.criticalVar.get(),
                    mask=self.maskVar.get(),
                    showmask=self.showmaskVar.get(),
                 )
        m,n = im.shape[:2]
        if m*n == 0:
            print( "Image Shape", im.shape, "Image cannot be set" )
            # raise Exception( "Simulator returns distorted image without any pixels." )
        else:
           if self.reflinesVar.get(): drawAxes(im)
           im0 = Image.fromarray( im )
           self.distorted.setImage(im0)
    def update(self):
        """
        Update the images with new data from the CosmoSim object.
        """
        self.setDistortedImage()
        self.setActualImage()
    def updateThread(self):
        while self._continue:
            self.updateEvent.wait()
            if self._continue:
               self.updateEvent.clear()
               self.update()
