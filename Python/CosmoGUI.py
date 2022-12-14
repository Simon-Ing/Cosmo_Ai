#! /usr/bin/env python3

"""
Desktop application to run the CosmoSim simulator for
gravitational lensing.

(Under construction.)
"""

from tkinter import *
from tkinter import ttk
from PIL import Image, ImageTk


from CosmoSim import CosmoSim

root = Tk()

sim = CosmoSim()
sim.init()
sim.runSim()



frm1 = ttk.Frame(root, padding=10)
frm2 = ttk.Frame(root, padding=10)
frm1.grid()
frm2.grid()
lensFrame = ttk.Frame(frm1, padding=10)
lensFrame.grid(column=0,row=1)
sourceFrame = ttk.Frame(frm1, padding=10)
sourceFrame.grid(column=1,row=1)

style = ttk.Style()
style.configure("Red.TButton", foreground="white", background="red")
labelstyle = ttk.Style()
labelstyle.configure("Std.TLabel", foreground="black", padding=4 )

quitButton = ttk.Button(frm1, text="Quit", command=root.destroy, style="Red.TButton")
quitButton.grid(column=2, row=0)

lensLabel = ttk.Label( lensFrame, text="Lens Model", style="Std.TLabel" )
lensSelector = ttk.Combobox( lensFrame, values=[ "SIS (roulettes)", "Point Mass (exact)", "Point Mass (roulettes)" ] )
lensLabel.grid(column=0, row=1, sticky=E )
lensSelector.grid(column=1, row=1)

einsteinLabel = ttk.Label( lensFrame, text="Einstein Radius", style="Std.TLabel" )
nTermsLabel = ttk.Label( lensFrame, text="Number of Terms (Roulettes only)", style="Std.TLabel" )
einsteinLabel.grid(column=0,row=3, sticky=E )
nTermsLabel.grid(column=0,row=4, sticky=E )

sourceLabel = ttk.Label( sourceFrame, text="Source Model", style="Std.TLabel" )
sourceSelector = ttk.Combobox( sourceFrame, values=[ "Spherical", "Ellipsoid", "Triangle" ] )
sourceLabel.grid(column=0, row=2, sticky=E )
sourceSelector.grid(column=1, row=2)

sigmaLabel = ttk.Label( sourceFrame, text="Source Size", style="Std.TLabel"  )
sigma2Label = ttk.Label( sourceFrame, text="Secondary Size", style="Std.TLabel"  )
thetaLabel = ttk.Label( sourceFrame, text="Source Rotation", style="Std.TLabel"  )
sigmaLabel.grid(column=0,row=3, sticky=E )
sigma2Label.grid(column=0,row=4, sticky=E )
thetaLabel.grid(column=0,row=5, sticky=E )



actual = Canvas(frm2,width=512,height=512)
actual.grid(column=0,row=0)
distorted = Canvas(frm2,width=512,height=512)
distorted.grid(column=1,row=0)

def setActualImage(im):
    im0 =  Image.fromarray(im)
    img =  ImageTk.PhotoImage(image=im0)
    im0.show()
    return actual.create_image(0,0,anchor=NW, image=img)
def setDistortedImage(im):
    im0 =  Image.fromarray(im)
    img =  ImageTk.PhotoImage(image=im0)
    return distorted.create_image(0,0,anchor=NW, image=img)

# img= ImageTk.PhotoImage(Image.open("test.png"))
# actual.create_image(0,0,anchor=NW,image=img)

# setDistortedImage(sim.getDistortedImage())
# setActualImage(sim.getActualImage())
im0 = Image.fromarray( sim.getActualImage() )
img0 = ImageTk.PhotoImage(image=im0)
actual.create_image(0,0,anchor=NW, image=img0)

im = Image.fromarray( sim.getDistortedImage() )
img = ImageTk.PhotoImage(image=im)
distorted.create_image(0,0,anchor=NW, image=img)

root.mainloop()

