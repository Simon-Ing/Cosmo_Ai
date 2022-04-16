
# Multiprocessing

import time
from sympy import lambdify, simplify, diff, symbols, sqrt
import sys
import concurrent.futures as fut
import pickle


def simpl(x):
    return simplify(x)


if __name__ == "__main__":
    n = int(sys.argv[1])
    timer = time.time()

    x, y = symbols('x, y', real=True)
    g, c = symbols("g c", positive=True, real=True)
    psi = (2*g)/(c**2)*sqrt(x**2 + y**2)
    alphas = [[0 for a in range(n)] for b in range(n)]
    betas = [[0 for c in range(n)] for d in range(n)]
    alphas_l = [[0 for e in range(n)] for f in range(n)]
    betas_l = [[0 for g in range(n)] for h in range(n)]


    # add base func
    alphas[0][1] = simpl(c * diff(psi, x))
    betas[0][1] = simpl(-c * diff(psi, y))
	
    # add diagonal
    for m in range(1, n-1):
        s = m + 1
        c = (m + 1.0)/(m + s + 1.0)*c
        print(f'm: {m}, s: {s} time: {time.time() - timer}')
        alphas[m][s] = c*(diff(alphas[m-1][s-1], x) - diff(betas[m-1][s-1], y))
        betas[m][s] = c*(diff(betas[m-1][s-1], x) + diff(alphas[m-1][s-1], y))


	
    # Add the rest
    start = 1
    for start in range(1, n, 2):
        for s in range(0, (n-start)):
            m = s + start
            # print(f'm: {m}, s: {s} time: {time.time() - timer}')
            c = ((m + 1.0)/(m + 1.0 - s) * (1.0 + (s != 0.0)) / 2.0)*c
            alphas[m][s] = c*(diff(alphas[m-1][s+1], x) + diff(betas[m-1][s+1], y))
            betas[m][s] = c*(diff(betas[m-1][s+1], x) - diff(alphas[m-1][s+1], y))

	
    # multiprocess simplify
    a_processes = []
    b_processes = []
    print("Simplifying")
    with fut.ProcessPoolExecutor() as executor:
        for m in range(1, n):
            for s in range((m+1)%2, min(m+2, n) , 2):
                a = executor.submit(simpl, alphas[m][s])
                b = executor.submit(simpl, betas[m][s])
                a_processes.append(a)
                b_processes.append(b)

        i = 0
        #print(f'tasks: {len(a_processes)}')
        for m in range(1, n):
            for s in range((m+1)%2, min(m+2, n) , 2):
                alphas[m][s] = a_processes[i].result()
                betas[m][s] = b_processes[i].result()
                # print(f'm: {m}, s: {s} time: {timer - time.time()}')
                print(alphas[m][s])
                i += 1

    # save symbolics
    print("Saving alphas")
    with open("alphas_" + str(n), "wb") as fp:
        pickle.dump(alphas, fp)

    print("Saving betas")

    with open("betas_" + str(n), "wb") as fp:
        pickle.dump(betas, fp)

    # Convert to string and save in txt
    for m in range(1, n):
        for s in range((m + 1) % 2, min(m + 2, n), 2):
            a_string = str(alphas[m][s])
            b_string = str(betas[m][s])

            with open('functions_sympy.txt', 'a') as f:
                f.write(f'{m}:{s}:{a_string}:{b_string}\n')



