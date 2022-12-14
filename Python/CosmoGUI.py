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



frm = ttk.Frame(root, padding=10)
# frm.grid()
l = ttk.Label(root, text="Hello World!")  # .grid(column=0, row=0)
b = ttk.Button(root, text="Quit", command=root.destroy)  # .grid(column=1, row=0)
l.pack()
b.pack()

actual = Canvas(root,width=512,height=512)  # .grid(column=0,row=1)
distorted = Canvas(root,width=512,height=512)  # .grid(column=1,row=1)
actual.pack()
distorted.pack()

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

