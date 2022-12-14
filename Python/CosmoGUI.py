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

l = ttk.Label(frm1, text="Hello World!").grid(column=0, row=0)
quitButton = ttk.Button(frm1, text="Quit", command=root.destroy, style="Red.TButton")
quitButton.grid(column=1, row=0)

lensLabel = ttk.Label( lensFrame, text="Lens Model" )
lensSelector = ttk.Combobox( lensFrame, values=[ "SIS (roulettes)", "Point Mass (exact)", "Point Mass (roulettes)" ] )
lensLabel.grid(column=0, row=1)
lensSelector.grid(column=1, row=1)

sourceLabel = ttk.Label( sourceFrame, text="Source Model" )
sourceSelector = ttk.Combobox( sourceFrame, values=[ "Spherical", "Ellipsoid", "Triangle" ] )
sourceLabel.grid(column=0, row=2)
sourceSelector.grid(column=1, row=2)



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

