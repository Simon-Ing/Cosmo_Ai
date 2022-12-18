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
        self.actual = Canvas(self,width=512,height=512)
        self.actual.grid(column=0,row=0)
        self.distorted = Canvas(self,width=512,height=512)
        self.distorted.grid(column=1,row=0)
        im = Image.fromarray( np.zeros((512,512)) )
        img =  ImageTk.PhotoImage(image=im)
        self.actualCanvas = self.actual.create_image(0,0,anchor=NW, image=img)
        self.distortedCanvas = self.distorted.create_image(0,0,
                anchor=NW, image=img)
        self.reflinesVar = BooleanVar()
        self.reflinesVar.set( True )

        self.updateEvent = sim.getUpdateEvent()
        self.updateThread = th.Thread(target=self.updateThread)
        self.updateThread.start()
    def getReflinesVar(self):
        return self.reflinesVar
    def close(self):
        """
        Terminate the update thread.
        """
        print ( "CosmoSim View object closing" )
        self._continue = False
        self.updateEvent.set()
        self.updateThread.join()
        print ( "CosmoSim View object closed" )
    def setActualImage(self):
        "Helper for `update()`."
        im0 = Image.fromarray( 
                self.sim.getActualImage( 
                    reflines=self.reflinesVar.get() ) )
        # Use an attribute to prevent garbage collection here
        self.img0 =  ImageTk.PhotoImage(image=im0)
        self.actual.itemconfig(self.actualCanvas, image=self.img0)
    def setDistortedImage(self):
        "Helper for `update()`."
        im1 = Image.fromarray( 
                self.sim.getDistortedImage( 
                    reflines=self.reflinesVar.get() ) )
        # Use an attribute to prevent garbage collection here
        self.img1 =  ImageTk.PhotoImage(image=im1)
        self.distorted.itemconfig(self.distortedCanvas, image=self.img1)
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
        print( "updateThread() returning" )
