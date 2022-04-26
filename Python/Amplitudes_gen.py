
import multiprocessing as mp
import sys
import time


from sympy import simplify, symbols, sqrt#, diff
from symengine import diff


n = int(sys.argv[1])

fn = str(n) + '.txt'


def listener(q):
    '''listens for messages on the q, writes to file. '''
    with open(fn, 'w') as f:
        while 1:
            # print(f'Jobs running: {}')
            m = q.get()
            # print("got write job:", m)
            if m == 'kill':
                f.write('killed')
                break
            f.write(str(m) + '\n')
            f.flush()


def simpl(x):
    return (x)


def func(n, m, s, alpha, beta, x, y, g, chi, q):
    print(f'm: {m} s: {s}')# alpha: {alpha} beta: {beta}')
    while s > 0 and m < n-1:
        m += 1
        s -= 1
        c = ((m + 1.0) / (m + 1.0 - s) * (1.0 + (s != 0.0)) / 2.0) * chi
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
    g, chi = symbols("g c", positive=True, real=True)
    psi = (2*g)/(chi**2) * sqrt(x ** 2 + y ** 2)
    # alphas = [[0 for a in range(n)] for b in range(n)]
    # betas = [[0 for c in range(n)] for d in range(n)]

    #must use Manager queue here, or will not work
    manager = mp.Manager()
    q = manager.Queue()
    with mp.Pool(processes=n) as pool:

        # use a separate process to write to file to avoid ksgjladsfkghldøf
        pool.apply_async(listener, (q,))

        jobs = []
        for m in range(0, n-1):
            # print(m)
            s = m + 1
            # print(m, s)
            if m == 0:
                alpha = simpl(-chi * diff(psi, x))
                beta = simpl(-chi * diff(psi, y))
            else:
                c = (m + 1.0) / (m + s + 1.0) * chi
                # calc sym func
                alpha_ = simpl(c * (diff(alpha, x) - diff(beta, y)))
                beta_ = simpl(c * (diff(beta, x) + diff(alpha, y)))
                alpha, beta = alpha_, beta_


            # print(f'{m}, {s} Done')
            res = f'{m}:{s}:{alpha}:{beta}'
            # print("send write job:", res)
            q.put(res)

            job = pool.apply_async(func, (n, m, s, alpha, beta, x, y, g, chi, q))
            jobs.append(job)



    # collect results from the workers through the pool result queue
        for job in jobs:
            job.get()

    #now we are done, kill the listener
    q.put('kill')
    pool.close()
    pool.join()

    print(time.time() - start)

if __name__ == "__main__":
   main()