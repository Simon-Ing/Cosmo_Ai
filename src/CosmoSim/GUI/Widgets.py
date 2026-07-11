#! /usr/bin/env python3
# (C) 2026: Hans Georg Schaathun <georg@schaathun.net> 

"""
A few widgets for internal use in the GUI submodule.
"""

from tkinter import *
from tkinter import ttk

class FloatSlider:
    """
    A slider for integer values with label.
    """
    def __init__(self,root,text,row=1,fromval=0,toval=100,var=None,
            resolution=0.01, default=0):
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
            self.var = DoubleVar()
        else:
            self.var = var
        self.var.set( default )
        self.label = ttk.Label( root, text=text,
                style="Std.TLabel" )
        self.slider = Scale( root, length=250, variable=self.var,
                resolution=resolution,
                digits=3,
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
