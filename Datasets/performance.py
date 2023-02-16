#! /usr/bin/env python2
# Generate a set of lens/source parameters

from random import randint
import numpy as np
import argparse



def getline(idx,chi=0,nterms=16):
    if 0 == chi:
        chi = randint(30,70)

    # Lens
    einsteinR = randint(5,50)

    # Polar Source Co-ordinates
    phi = randint(0,359)
    R = randint(einsteinR,100)

    # Source
    sigma = randint(1,round((200-einsteinR)/3))
    sigma2 = 0
    theta = 0

    # Cartesian Co-ordinates
    x = R*np.cos(np.pi*phi/180)
    y = R*np.sin(np.pi*phi/180)

    srcmode = "s"
    lensmode = "s"
    return f'"{idx:06}",image-{idx:06}.png,{srcmode},{lensmode},{chi},' \
         + f'{R},{phi},{einsteinR},{sigma},{sigma2},{theta},{nterms},{x},{y}'

header = ( "index,filename,source,lens,chi,"
         + "R,phi,einsteinR,sigma,sigma2,theta,nterms,x,y\n"
         )

def main():
    fn = "performance.csv"
    parser = argparse.ArgumentParser(
            prog = 'CosmoSim data set generator',
            description = 'Generate images for training and testing',
            epilog = '')

    parser.add_argument('-o','--outfile',help="Output file")
    parser.add_argument('-f', '--start',
            help="First image number to generate")
    parser.add_argument('-n', '--count',
            help="Number of images image to generate")
    parser.add_argument('-m', '--nterms',
            help="Truncation point")
    args = parser.parse_args()
    if args.nterms:
        nterms = int(args.nterms)
    else:
        nterms = 50
    if args.start:
        idxstart = int(args.start)
    else:
        idxstart = 0
    if args.count:
        idxend = idxstart + int(args.count)
    else:
        idxend = idxstart + 1000
    if args.outfile:
        fn = args.outfile

    with open(fn, 'w') as f:
      f.write(header)
      for i in range(idxstart,idxend):
        l = getline(i+1,chi=50,nterms=nterms)
        f.write(l)
        f.write("\n")

if __name__ == "__main__":
   main()
