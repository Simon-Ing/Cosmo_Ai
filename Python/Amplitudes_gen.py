#! /usr/bin/env python3

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

from sympy import simplify, symbols, sqrt, diff, factor

# from symengine import diff


if len(sys.argv) < 2:
   n = 50
else:
   n = int(sys.argv[1])
if len(sys.argv) < 3:
   nproc = 50
else:
   nproc = int(sys.argv[2])

fn = str(n) + '.txt'

def listener(q):
    '''listens for messages on the q, writes to file. '''
    with open(fn, 'w') as f:
        while 1:
            # print(f'Jobs running: {}')
            m = q.get()
            # print("got write job:", m)
            if m == 'kill':
                print("Done")
                break
            f.write(str(m) + '\n')
            f.flush()


def simpl(x):
    return factor(x)


def func(n, m, s, alpha, beta, x, y, g, q):
    print(f'm: {m} s: {s}')# alpha: {alpha} beta: {beta}')
    while s > 0 and m < n:
        m += 1
        s -= 1
        c = ((m + 1.0) / (m + 1.0 - s) * (1.0 + (s != 0.0)) / 2.0)
        # start calculate
        alpha_ = simpl(c * (diff(alpha, x) + diff(beta, y)))
        beta_ = simpl(c * (diff(beta, x) - diff(alpha, y)))
        alpha, beta = alpha_, beta_
        print(f'm: {m} s: {s}')# alpha: {alpha} beta: {beta} c: {c}')
        # print("gen write job")
        res = f'{m}:{s}:{alpha}:{beta}'
        # print(f'{m}, {s} Done')
        q.put(res)

def main():

    global num_processes

    start = time.time()


    x, y = symbols('x, y', real=True)
    g = symbols("g", positive=True, real=True)
    psi = - (2*g) * sqrt(x ** 2 + y ** 2)

    #must use Manager queue here, or will not work
    manager = mp.Manager()
    q = manager.Queue()
    with mp.Pool(processes=nproc) as pool:

        # use a separate process to write to file to avoid ksgjladsfkghldøf
        pool.apply_async(listener, (q,))

        jobs = []
        for m in range(0, n+1):
            # print(m)
            s = m + 1
            # print(m, s)
            if m == 0:
                alpha = simpl(diff(psi, x))
                beta = simpl(diff(psi, y))
            else:
                c = (m + 1.0) / (m + s + 1.0) 
                # calc sym func
                alpha_ = simpl(c * (diff(alpha, x) - diff(beta, y)))
                beta_ = simpl(c * (diff(beta, x) + diff(alpha, y)))
                alpha, beta = alpha_, beta_


            # print(f'{m}, {s} Done')
            res = f'{m}:{s}:{alpha}:{beta}'
            # print("send write job:", res)
            q.put(res)

            job = pool.apply_async(func, (n, m, s, alpha, beta, x, y, g, q))
            jobs.append(job)



    # collect results from the workers through the pool result queue
        for job in jobs:
            job.get()

    #now we are done, kill the listener
    q.put('kill')
    print( "Issued kill signal" )
    pool.close()
    print( "Pool closed" )
    pool.join()
    print( "Pool joined" )

    print(time.time() - start)

if __name__ == "__main__":
   main()
