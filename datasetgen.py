#! /usr/bin/env python2
# Generate a set of lens/source parameters

import multiprocessing as mp
import sys
import time


from random import randint

n = 40

fn = "dataset.csv"
srcmodes = "e"
lensmodes = "pss"


def getline(idx,chi=0,nterms=16):
    if 0 == chi:
        chi = randint(30,70)
    x = randint(-75,75)
    y = randint(-75,75)
    einsteinR = randint(10,50)
    sigma = randint(1,60)
    sigma2 = randint(1,40)
    theta = randint(0,179)
    srcmode = srcmodes[randint(0,len(srcmodes)-1)]
    lensmode = lensmodes[randint(0,len(lensmodes)-1)]
    return f'{idx:04},{srcmode},{lensmode},{chi},{x},{y},{einsteinR},{sigma},{sigma2},{theta},{nterms}'


def main():
    with open(fn, 'w') as f:
      for i in range(n):
        l = getline(i+1)
        f.write(l)
        f.write("\n")

if __name__ == "__main__":
   main()
