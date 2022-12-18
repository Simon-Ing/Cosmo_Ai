#! /usr/bin/env python3
# (C) 2022: Hans Georg Schaathun <georg@schaathun.net> 

"""
Desktop application to run the CosmoSim simulator for
gravitational lensing.
"""

from tkinter import *
from tkinter import ttk

from CosmoSim import CosmoSim
import CosmoSim.Controller as cont
import CosmoSim.View as view

class Window(Tk):
    """
    The main application window.
    """
    def destroy(self):
        self.sim.close()
        self.imgPane.close()
        return super().destroy()
    def __init__(self,sim,*a,**kw):
        super().__init__(*a,**kw)
        self.sim = sim

        ttk.Style().configure("Red.TButton",
                foreground="white", background="red")
        ttk.Style().configure("Std.TLabel", foreground="black", padding=4,
                font=( "Arial", 15 ) )
        ttk.Style().configure("Std.TCheckbutton", foreground="black", 
                relief=[ ( "selected", "sunken" ), ( "!selected", "raised" ) ],
                 )

        controller = cont.Controller(self,sim, padding=10)
        controller.grid()
        self.imgPane = view.ImagePane(self,sim, padding=10)
        self.imgPane.grid()
        self.frm = ttk.Frame(self, padding=10)
        self.frm.grid()
        self.quitButton = ttk.Button(self.frm, text="Quit",
                command=self.destroy, style="Red.TButton" )
        self.quitButton.grid(column=4, row=0, sticky=E)
        Label(self.frm,"" , width=20 ).grid(column=2,row=0)
        self.reflineButton = ttk.Checkbutton(self.frm,
                onvalue=True, offvalue=False,
                variable=self.imgPane.getReflinesVar(),
                text="Show Reference Lines" )
        self.reflineButton.grid(column=0, row=0, sticky=E)
        self.maskButton = ttk.Checkbutton(self.frm,
                onvalue=True, offvalue=False,
                variable=self.imgPane.getMaskVar(),
                text="Mask Roulettes Image" )
        self.maskButton.grid(column=1, row=0, sticky=E)
        self.showmaskButton = ttk.Checkbutton(self.frm,
                onvalue=True, offvalue=False,
                variable=self.imgPane.getShowmaskVar(),
                text="Show Masks" )
        self.showmaskButton.grid(column=2, row=0, sticky=E)

if __name__ == "__main__":
    print( "CosmoGUI starting." )

    sim = CosmoSim()
    win = Window(sim)
    sim.runSimulator()

    win.mainloop()

