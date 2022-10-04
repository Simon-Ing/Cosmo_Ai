#! /usr/bin/env python3
# Generate a set of lens/source parameters

import multiprocessing as mp
import sys
import time


from random import randint

n = 300

fn = "dataset.csv"

def getline(idx,chi=0,nterms=16):
    if 0 == chi:
        chi = randint(30,70)
    srcmode = "s"
    lensmode = "p"
    x = randint(0,100)
    y = randint(0,100)
    einsteinR = randint(0,50)
    sigma = randint(1,50)
    sigma2 = randint(1,50)
    theta = randint(0,179)
    return f'{idx:04},{srcmode},{lensmode},{x},{y},{einsteinR},{sigma},{sigma2},{theta},{nterms}'


def main():
    with open(fn, 'w') as f:
      for i in range(n):
        l = getline(i+1)
        f.write(l)
        f.write("\n")

if __name__ == "__main__":
   main()
