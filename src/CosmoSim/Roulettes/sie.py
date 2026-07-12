#! /usr/bin/env python3
# (C) 2024: Hans Georg Schaathun <georg@schaathun.net>

"""
Symbolically differentiate the lens potential of the SIE lens.

- `n` is the maximum number of terms
- `nproc` is the number of threads
"""

import multiprocessing as mp
import sys
import os
import time
import argparse
import queue

import sympy
from .libamplitudes import *
from sympy import symbols, diff, sin, cos, asin, atan2, asinh, binomial

def firstworker(q,resDict,maxm=6):
    print ( os.getpid(),"working" )
    while True:
        i,j,psi,x,y = q.get(True)   # Blocks until there is a job on the queue
        if i == 0:
           res = sympy.simplify( diff( psi, y ) )
        else:
           res = sympy.simplify( diff( psi, x ) )
        resDict[(i,j)] = res
        if i+j <= maxm:     # Submit jobs for next round
            q.put( (i+1, j, res, x, y) ) 
            if i==0:
               q.put( (0, j+1, res, x, y) ) 
        print( "I.", os.getpid(), i, j )
        q.task_done()     # Tell the queue that the job is complete

        # Note that new jobs are submitted to the queue before 
        # `q.task_done()` is called.  Hence the queue will not be
        # empty if the current job should spawn new ones. 
    print ( "I.", os.getpid(),"returning" )
def secondworker(q,psidiff,diff1,vars):
    print ( "secondworker", os.getpid(),"working" )
    cont = True
    theta = symbols("p",real=True)
    x,y = vars
    x1,y1 = symbols("x1 y1",real=True)
    while cont:
      try:
        m,n = q.get(False)   # does not block
        res = sum( [
                  binomial( m, i )*
                  binomial( n, j )*
                  cos(theta)**(m-i+j)*
                  sin(theta)**(n-j+i)*
                  (-1)**i
                  * diff1[(m+n-i-j,j+i)]
                  for i in range(m+1) for j in range(n+1) ] ) 
        res = res.subs([ ( x, x1 ), ( y, y1 ) ])
        res = res.subs([ ( x1, cos(theta)*x+sin(theta)*y ),
                   ( y1, -sin(theta)*x+cos(theta)*y ) ] )
        # res = sympy.expand( res )  # This is too slow
        # res = sympy.simplify( res )  # This is too slow
        print( "II.", os.getpid(), m, n )
        psidiff[(m,n)] = res
      except queue.Empty:
        print ( "II.", os.getpid(), "completes" )
        cont = False
    print ( "II.", os.getpid(),"returning" )
def cworker(q,psidiff,diff1,vars):
    print ( "cworker", os.getpid(),"working" )
    cont = True
    x,y = vars
    while cont:
      try:
        m,n = q.get(False)   # does not block
        res = sum( [
                  binomial( m, i )*
                  binomial( n, j )*
                  1**(m-i+j)*
                  0**(n-j+i)*
                  (-1)**i
                  * diff1[(m+n-i-j,j+i)]
                  for i in range(m+1) for j in range(n+1) ] ) 
        res = sympy.expand( res )    # This is too slow
        # res = sympy.simplify( res )  # This is too slow
        print( "II. (circular)", os.getpid(), m, n )
        psidiff[(m,n)] = res
      except queue.Empty:
        print ( "II. (circular)", os.getpid(), "completes" )
        cont = False
    print ( "II. (circular)", os.getpid(),"returning" )

def thirdworker(q,ampdict,indict, var=[] ):
    print ( os.getpid(),"working" )
    cont = True
    while cont:
      try:
        m,s = q.get(False)   # does not block
        a = - sympy.collect( sum( [
                  binomial( m, k ) *
                  ( cfunc(m,k,s)*indict[(m-k+1,k)]
                  + cfunc(m,k+1,s)*indict[(m-k,k+1)] )
                  for k in range(m+1) ] ),
                  var )
        b = - sympy.collect( sum( [
                  binomial( m, k ) *
                  ( sfunc(m,k,s)*indict[(m-k+1,k)]
                  + sfunc(m,k+1,s)*indict[(m-k,k+1)] )
                  for k in range(m+1) ] ),
                  var )
        if s == 0:
               a /= 2
        print( "III (Chris).", os.getpid(), m, s )
        sys.stdout.flush()
        ampdict[(m,s)] = (a,b)
      except queue.Empty:
        print ( "III.", os.getpid(), "completes" )
        cont = False

    print ( "III.", os.getpid(),"returning" )
    sys.stdout.flush()

class RouletteManager():
    def __init__(self,psivec=None,secondworker=secondworker,thirdworker=thirdworker):
       self.mgr = mp.Manager()      
       if psivec == None:
           self.psivec = psiSIE()
       else:
           self.psivec = psivec
       self.secondworker = secondworker
       self.thirdworker = thirdworker

    def getDict(self,n=50,nproc=None):

        q = mp.JoinableQueue()  # Input queue.  It is joinable to control completion.
        resDict = self.mgr.dict()     # Output data structure

        # Get and store the initial case m+s=1
        (psi,a,b,x,y) = self.psivec
        self.vars = x,y
        resDict[(0,0)] = psi
    
        # Submit first round of jobs
        q.put( (1,0,psi,x,y) )
        q.put( (0,1,psi,x,y) )

        # Create a pool of workers to process the input queue.
        with mp.Pool(nproc, firstworker,(q,resDict,n,)) as pool: 

            q.join()   # Block until the input jobs are completed
            print( "I getDict() joined queue" )

        self.diff1 = resDict

        print( "I getDict returns" )
        sys.stdout.flush()
        return resDict, x, y
    def getDiff(self,n,nproc):
        q = mp.Queue()          # Input queue
        psidiff = self.mgr.dict()     # Output data structure

        for k in self.diff1.keys():
           q.put( k )
        # All jobs are submitted before the Pool starts
        pool = mp.Pool(nproc, self.secondworker,(q,psidiff,self.diff1,self.vars))

        pool.close()
        pool.join()

        self.psidiff = psidiff

        print( "II. getDiff returns")
        sys.stdout.flush()
        return psidiff
    def getAmplitudes(self,n,nproc):

        start = time.time()
        self.getDict(n,nproc)
        print( "Time spent:", time.time() - start)

        self.getDiff(n,nproc)
        print( "Time spent:", time.time() - start)

        q = mp.Queue()         # Input queue
        rdict = self.mgr.dict()     # Output data structure
        rdict[(0,1)] = (-self.psidiff[1,0],
                        -self.psidiff[0,1] )

        for m in range(1,n+1):
           for s in range((m+1)%2,m+2,2):
               q.put((m,s))
        pool = mp.Pool(nproc, self.thirdworker,(q,rdict,self.psidiff,self.vars))

        pool.close()
        pool.join()

        print( "III getAmplitudes returns" )
        sys.stdout.flush()
        self.rdict = rdict
        print( "Time spent:", time.time() - start)
        return rdict


def main(f=thirdworker):
    parser = argparse.ArgumentParser(description='Generate roulette amplitude formulæ for CosmoSim.')
    parser.add_argument('n', metavar='N', type=int, nargs="?", default=50,
                    help='Max m (number of terms)')
    parser.add_argument('nproc', type=int, nargs="?",
                    help='Number of processes.')
    parser.add_argument('--output', help='Output filename')
    parser.add_argument('--diff', default=False,action="store_true",
                    help='Simply differentiate psi')
    parser.add_argument('--lens', default="SIE",
                    help='Lens model')
    parser.add_argument('--circular', default=False,action="store_true",
                    help='Circular (symmetric) lens')
    parser.add_argument('--tex', 
                    help='TeX output file')
    args = parser.parse_args()

    n = args.n
    if args.nproc is None: 
        nproc = n
    else:
        nproc = args.nproc
    print( f"Using {nproc} CPUs" )

    psivec = zeroth( args.lens )
    
    if args.circular:
        mgr = RouletteManager( psivec=psivec,secondworker=cworker,thirdworker=f )
    else:
        mgr = RouletteManager( psivec=psivec,thirdworker=f )
    alphabeta = mgr.getAmplitudes(n,nproc )

    if args.output:
       ampPrint(alphabeta,args.output)
    if args.tex:
       texPrint(alphabeta,args.tex,latex=sympy.latex,opt="10pt,paper=a0,landscape")
    print( "sieamplitudes.py: completing" )

if __name__ == "__main__":
    main()
