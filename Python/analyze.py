import matplotlib.pyplot as plt
import matplotlib
from numpy import arctan2

matplotlib.use('TkAgg')
from math import factorial, cos, sin, pi, sqrt
import dill
dill.settings['recurse'] = True

from sympy import symbols
import numpy as np



x, y, g, c = symbols("x y g c")

n = 20

size = 100


# alphas = [[0 for i in range(n)]for j in range(n)]
# betas = [[0 for i in range(n)]for j in range(n)]
#
# with open('../functions_20.txt') as file:
#     lines = file.readlines()
#     for line in lines:
#         i, j, alpha_, beta_ = line.split(":")
#         alphas[int(i)][int(j)] = lambdify((x, y, g, c), parse_expr(alpha_))
#         betas[int(i)][int(j)] = lambdify((x, y, g, c), parse_expr(beta_))
#
# with open("alphas", "wb") as outfile:
#     dill.dump(alphas, outfile)
#
# with open("betas", "wb") as outfile:
#     dill.dump(betas, outfile)

with open("alphas", "rb") as infile:
    alphas = dill.load(infile)

with open("betas", "rb") as infile:
    betas = dill.load(infile)

# alphas, betas = multi(n)




img = np.zeros((size, size))
dist = np.zeros((size, size))
CHI = 1
GAMMA = 1
X = 10
Y = 0

# for row in range(100):
#     for col in range(100):
#         x = col - 50 - app_pos
#         y = 50 - row

r = 10
theta = 1
ksi1 = 0
ksi2 = 0
ksi1s_ = []
ksi2s_ = []

fig = plt.figure(figsize=(10, 8))

for i, r in enumerate([5, 10, 15, 20]):
    ksi1s = []
    ksi2s = []
    k = 0
    for m in range(1, n):
        frac = r**m / factorial(m)
        sub_term1 = 0
        sub_term2 = 0
        for s in range((m+1) % 2, min(m+2, n), 2):
            alpha = alphas[m][s](X, Y, CHI, GAMMA)
            beta = betas[m][s](X, Y, CHI, GAMMA)
            c_p = 1 + s/(m + 1)
            c_m = 1 - s/(m + 1)
            sub_term1 += 1/2*((alpha*cos((s-1)*theta) + beta*sin((s-1)*theta)*c_p + (alpha*cos((s+1)*theta) + beta*sin(s+1)*theta))*c_m)
            sub_term2 += 1/2*((-alpha*sin((s-1)*theta) + beta*cos((s-1)*theta)*c_p + (alpha*sin((s+1)*theta) - beta*cos(s+1)*theta))*c_m)
            term1 = frac*sub_term1
            term2 = frac*sub_term2
            ksi1 += term1
            ksi2 += term2
            ksi1s.append(ksi1)
            ksi2s.append(ksi2)
    ax = plt.subplot(2, 2, i+1)
    # ax.title.set_text(f'theta = {theta}')
    t = [i for i, _ in enumerate(ksi1s)]
    ax.title.set_text(f'r = {r}')
    plt.plot(t, ksi1s, label="ksi 1")
    plt.plot(t, ksi2s, 'r', label="ksi 2")
    plt.legend()
    plt.grid()

title = f'theta = 1'
fig.suptitle(title, fontsize=16)
# plt.savefig("r = 10+-")
plt.show()


#     double frac = pow(r, m) / factorial_(m);
# double subTerm1 = 0;
# double subTerm2 = 0;
# for (int s=(m+1)%2; s<=m+1 && s<n; s+=2){
# double alpha = alphas_lambda[m][s].call({X, Y, GAMMA, CHI});
# double beta = betas_lambda[m][s].call({X, Y, GAMMA, CHI});
# int c_p = 1 + s/(m + 1);
# int c_m = 1 - s/(m + 1);
# subTerm1 += theta*((alpha*cos(s-1) + beta*sin(s-1))*c_p + (alpha*cos(s+1) + beta*sin(s+1)*c_m));
# subTerm2 += theta*((-alpha*sin(s-1) + beta*cos(s-1))*c_p + (alpha*sin(s+1) - beta*cos(s+1)*c_m));
# //            std::cout << "\nm: " << m << " s: " << s << "\nsubterm1: " << subTerm1 << "\nsubterm2: " << subTerm2 << std::endl;
# }
# double term1 = frac*subTerm1;
# double term2 = frac*subTerm2;
# //        std::cout << "\nm: " << m << "\nterm1: " << term1 << "\nterm2: " << term2 << "\nksi1: " << ksi1 << "\nksi2: " << ksi2 << std::endl;
# ksi1 += term1;
# ksi2 += term2;




