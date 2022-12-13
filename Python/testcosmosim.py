import CosmoSim as cs
import matplotlib.pyplot as plt

sim = cs.CosmoSim()
sim.init()
sim.runSim()
sim.diagnostics() 
im = sim.getDistortedImage()
print( "Image size in python:", im.shape, im.dtype )
im.shape = im.shape[:2]
print ( im.shape )
plt.imsave("testcosmosim.png",im,cmap="gray")

im = sim.getActualImage()
im.shape = im.shape[:2]
plt.imsave("testcosmosim-actual.png",im,cmap="gray")
