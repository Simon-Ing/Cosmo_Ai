#! /usr/bin/env python2
# Generate a set of lens/source parameters

import multiprocessing as mp
import sys
import time
import numpy as np


from random import randint

n = 300

fn = "dataset.csv"
srcmodes = "e"
lensmodes = "pss"


def getline(idx,chi=0,nterms=16):
    if 0 == chi:
        chi = randint(30,70)

    # Source
    sigma = randint(1,60)
    sigma2 = randint(1,40)
    theta = randint(0,179)

    # Lens
    einsteinR = randint(10,50)

    # Polar Source Co-ordinates
    phi = randint(0,359)
    R = randint(einsteinR,100)

    # Cartesian Co-ordinates
    x = R*np.cos(np.pi*phi/180)
    y = R*np.sin(np.pi*phi/180)

    srcmode = srcmodes[randint(0,len(srcmodes)-1)]
    lensmode = lensmodes[randint(0,len(lensmodes)-1)]
    return f'"{idx:04}",image-{idx:04}.png,{srcmode},{lensmode},{chi},' \
         + f'{R},{phi},{einsteinR},{sigma},{sigma2},{theta},{nterms},{x},{y}'

header = ( "index,filename,source,lens,chi,"
         + "R,phi,einsteinR,sigma,sigma2,theta,nterms,x,y\n"
         )

def main():
    with open(fn, 'w') as f:
      f.write(header)
      for i in range(n):
        l = getline(i+1)
        f.write(l)
        f.write("\n")

if __name__ == "__main__":
   main()
