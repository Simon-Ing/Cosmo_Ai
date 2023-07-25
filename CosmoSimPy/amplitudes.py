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

from sympy import simplify, symbols, sqrt, diff, factor, sin, cos, asin, atan2, asinh

def listener(fn,q):
    '''Listens for messages on the Queue q and writes to file `fn`. '''
    print( "Listener starts with file ", fn ) 
    with open(fn, 'w') as f:
        print( "Opened file", fn ) 
        while 1:
            # print(f'Jobs running: {}')
            m = q.get()
            # print("got write job:", m)
            if m == 'kill':
                print("Done")
                break
            f.write(str(m) + '\n')
            f.flush()
        print( "File writer terminated", fn ) 
    if hit_except: print( "Failed to open file ", fn )

def func(n, m, s, alpha, beta, x, y, q):
    """
    Generate the amplitudes for one fixed sum $m+s$.
    This is done by recursion on s and m.
    """
    print(f'm: {m} s: {s}')# alpha: {alpha} beta: {beta}')
    while s > 0 and m < n:
        m += 1
        s -= 1
        c = ((m + 1.0) / (m + 1.0 - s) * (1.0 + (s != 0.0)) / 2.0)
        # start calculate
        alpha_ = factor(c * (diff(alpha, x) + diff(beta, y)))
        beta_ = factor(c * (diff(beta, x) - diff(alpha, y)))
        alpha, beta = alpha_, beta_
        print(f'm: {m} s: {s}') # alpha: {alpha} beta: {beta} c: {c}')

        res = f'{m}:{s}:{alpha}:{beta}'
        # print ( "Inner", res )
        q.put(res)

def psiSIS():
    # g is the Einstein radius and (x,y) coordinates in the lens plane
    x, y = symbols('x, y', real=True)
    g = symbols("g", positive=True, real=True)
    psi = - g * sqrt(x ** 2 + y ** 2)
    return (psi,x,y)
def psiSIE():
    # g is the Einstein radius and (x,y) coordinates in the lens plane
    x, y = symbols('x, y', real=True)
    g = symbols("g", positive=True, real=True)
    f = symbols("f", positive=True, real=True)
    p = symbols("p", positive=True, real=True)
    psi = - g * sqrt(x ** 2 + y ** 2) * sqrt( f/(1-f*f) )  * (
            sin( p + atan2(y,x) ) * asin( sqrt( 1-f*f )* sin( p + atan2(y,x) ) )
            + cos( p + atan2(y,x) ) * asinh( sqrt( 1-f*f )/f* cos( p + atan2(y,x) ) )
            )
    return (psi,x,y)
def main(psi=None,n=50,nproc=None,fn=None):

    global num_processes

    if nproc is None: nproc = n

    
    # The filename is generated from the number of amplitudes
    if fn is None: fn = str(n) + '.txt'

    start = time.time()


    # psi is the lens potential
    if psi is None:
       psi, x, y = psiSIS()
    else:
       psi, x, y = psi

    # Must use Manager queue here, or will not work
    manager = mp.Manager()
    q = manager.Queue()

    with mp.Pool(processes=nproc) as pool:

        # use a single, separate process to write to file 
        pool.apply_async(listener, (fn,q,))

        jobs = []
        for m in range(0, n+1):

            s = m + 1

            if m == 0:
                # This is the base case (m,s)=(0,1) of the outer recursion
                alpha = factor(diff(psi, x))
                beta = factor(diff(psi, y))
            else:
                # This is the base case (m+1,s+1) of the inner recursion
                c = (m + 1.0) / (m + s + 1.0) 
                # Should there not be an extra factor 2 for s==1 above?
                # - maybe it does not matter because s=m+1 and m>1.
                alpha_ = factor(c * (diff(alpha, x) - diff(beta, y)))
                beta_ = factor(c * (diff(beta, x) + diff(alpha, y)))
                alpha, beta = alpha_, beta_


            res = f'{m}:{s}:{alpha}:{beta}'
            # print ( "Outer", res )
            q.put(res)

            job = pool.apply_async(func, (n, m, s, alpha, beta, x, y, q))
            jobs.append(job)

        # collect results from the workers through the pool result queue
        for job in jobs:
            job.get()

    # Now we are done, kill the listener
    q.put('kill')
    print( "Issued kill signal" )
    pool.close()
    print( "Pool closed" )
    pool.join()
    print( "Pool joined" )

    print( "Time spent:", time.time() - start)

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

    args = parser.parse_args()
    if args.lens == "SIS":
        model = psiSIS()
    elif args.lens == "SIE":
        model = psiSIE()
    else:
        model = None
    if args.diff:
        psi,x,y = model
        dx = factor(diff(psi, x))
        dy = factor(diff(psi, y))
        print( "dx", dx )
        print( "dy", dy )
    else:
        main(psi=model,n=args.n,nproc=args.nproc,fn=args.output)
