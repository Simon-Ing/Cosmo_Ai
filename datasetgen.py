#! /usr/bin/env python2
# Generate a set of lens/source parameters

import multiprocessing as mp
import sys
import time


from random import randint

n = 100

fn = "dataset.csv"


def getline(idx,chi=0,nterms=16):
    if 0 == chi:
        chi = randint(30,70)
    x = randint(-75,75)
    y = randint(-75,75)
    einsteinR = randint(10,50)
    sigma = randint(1,60)
    sigma2 = randint(1,40)
    theta = randint(0,179)
    srcmode = "sse"[randint(0,2)]
    lensmode = "pss"[randint(0,2)]
    return f'{idx:04},{srcmode},{lensmode},{chi},{x},{y},{einsteinR},{sigma},{sigma2},{theta},{nterms}'


def main():
    with open(fn, 'w') as f:
      for i in range(n):
        l = getline(i+1)
        f.write(l)
        f.write("\n")

if __name__ == "__main__":
   main()
