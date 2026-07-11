#! /usr/bin/env python3
# (C) 2023: Hans Georg Schaathun <georg@schaathun.net>

"""
Generate the 50.txt file containing expressions for alpha and beta
for the SIS model in Roulettes.

Usage: `python3 Amplitudes_gen.py [n [nproc]]`

- `n` is the maximum number of terms
- `nproc` is the number of threads
"""

import multiprocessing as mp
import sys
import time
import argparse

import sympy
from .libamplitudes import *
from sympy import symbols, sqrt, diff, sin, cos, asin, atan2, asinh, Rational

def func(n, m, s, alpha, beta, x, y, rdict):
    """
    Generate the amplitudes for one fixed sum $m+s$.
    This is done by recursion on s and m.
    """
    print(f'm: {m} s: {s}')# alpha: {alpha} beta: {beta}')
    while s > 0 and m < n:
        m += 1
        s -= 1
        if s == 0:
            c = Rational(1,2)
        else:
            c = Rational(m + 1, m + 1 - s) 
        # start calculate
        alpha_ = sympy.factor(c * (diff(alpha, x) + diff(beta, y)))
        beta_ = sympy.factor(c * (diff(beta, x) - diff(alpha, y)))
        alpha, beta = alpha_, beta_
        print(f'm: {m} s: {s}') # alpha: {alpha} beta: {beta} c: {c}')

        # res = f'{m}:{s}:{alpha}:{beta}'
        # print ( "Inner", res )
        # q.put(res)
        rdict[(m,s)] = (alpha,beta) 


def main(lens="SIS",n=50,nproc=None,fn=None):

    global num_processes

    if nproc is None: nproc = n
    print( f"Using {nproc} CPUs" )

    
    # The filename is generated from the number of amplitudes
    if fn is None: fn = str(n) + '.txt'

    start = time.time()

    # Must use Manager queue here, or will not work
    manager = mp.Manager()
    # q = manager.Queue()
    rdict = manager.dict()

    with mp.Pool(processes=nproc) as pool:

        # use a single, separate process to write to file 
        # pool.apply_async(listener, (fn,q,))

        jobs = []
        for m in range(0, n+1):

            s = m + 1

            if m == 0:
                # This is the base case (m,s)=(0,1) of the outer recursion
                psi, alpha, beta, x, y = zeroth(lens)
                alpha = - alpha
                beta = - beta
            else:
                # This is the base case (m+1,s+1) of the inner recursion
                c = Rational(m + 1, m + 1 + s) 
                # Should there not be an extra factor 2 for s==1 above?
                # - maybe it does not matter because s=m+1 and m>1.
                alpha_ = sympy.factor(c * (diff(alpha, x) - diff(beta, y)))
                beta_ = sympy.factor(c * (diff(beta, x) + diff(alpha, y)))
                alpha, beta = alpha_, beta_


            res = f'{m}:{s}:{alpha}:{beta}'
            # print ( "Outer", res )
            rdict[(m,s)] = (alpha,beta) 

            job = pool.apply_async(func, (n, m, s, alpha, beta, x, y, rdict))
            jobs.append(job)

        # collect results from the workers through the pool result queue
        for job in jobs:
            job.get()

    # Now we are done, kill the listener
    # q.put('kill')
    print( "[amplitudes.py]  Completed.  Issued kill order to terminate." )
    pool.close()
    print( "Pool closed" )
    pool.join()
    print( "Pool joined" )

    print( "Time spent:", time.time() - start)
    return rdict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate roulette amplitude formulæ for CosmoSim.')
    parser.add_argument('n', metavar='N', type=int, nargs="?", default=50,
                    help='Max m (number of terms)')
    parser.add_argument('nproc', type=int, nargs="?",
                    help='Number of processes.')
    parser.add_argument('--lens', default="SIS",
                    help='Lens model')
    parser.add_argument('--output', help='Output filename')
    parser.add_argument('--diff', default=False,action="store_true",
                    help='Simply differentiate psi')
    parser.add_argument('--tex', 
                    help='TeX output file')
    args = parser.parse_args()

    if args.diff:
        psi, dx,dy,x,y = zeroth(args.lens)
        print( "dx", dx )
        print( "dy", dy )
    else:
        alphabeta = main(lens=args.lens,n=args.n,nproc=args.nproc,fn=args.output)
        if args.output:
           ampPrint(alphabeta,args.output)
        elif args.tex:
           texPrint(alphabeta,args.tex)
        else:
            ampPrint(alphabeta)
