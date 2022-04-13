from math import factorial
from time import time
from sympy import lambdify, symbols
from matplotlib import pyplot as plt
from numpy import exp, cos, sin, pi, sqrt, arctan2
import numpy as np
import dill
import cv2 as cv


def spherical(r, theta, X, Y, GAMMA, CHI):
    ksi1 = 0
    ksi2 = 0

    count = 0
    max1 = 0
    max2 = 0

    for m in range(1, n):
        frac = pow(r, m) / factorial(m)
        sub_term1 = 0.0
        sub_term2 = 0.0
        for s in range((m+1)%2, min(m+1, n), 2):
            count += 1
            alpha = alphas[m][s](X, Y, GAMMA, CHI)
            beta = betas[m][s](X, Y, GAMMA, CHI)
            c_p = 1 + s/(m + 1)
            c_m = 1 - s/(m + 1)
            sub_term1 += theta*((alpha*cos(s-1) + beta*sin(s-1))*c_p + (alpha*cos(s+1) + beta*sin(s+1)*c_m))
            sub_term2 += theta*((-alpha*sin(s-1) + beta*cos(s-1))*c_p + (alpha*sin(s+1) - beta*cos(s+1)*c_m))

        term1 = frac*sub_term1
        term2 = frac*sub_term2
        ksi1 += term1
        ksi2 += term2

        print(f'm: {m}, term1: {term1}, term2: {term2} frac: {frac}')

    return ksi1, ksi2


if __name__ == "__main__":
    # x, y = symbols('x, y', real=True)
    # gamma, chi = symbols("gamma chi", positive=True, real=True)

    print("Loading alphas")
    with open("alphas_l_14", "rb") as fp:
        alphas = dill.load(fp)

    print("Loading betas")
    with open("betas_l_14", "rb") as fp:
        betas = dill.load(fp)
    print("Done loading")

    n = len(alphas)
    print(f'n: {n}')

    size = 100
    x_pos = 10
    y_pos = 10
    GAMMA = 100
    CHI = 0.5
    X = x_pos * CHI
    Y = y_pos * CHI

    img = np.zeros((size, size))
    dist = np.zeros((size, size))

    img = cv.circle(img, (int(size/2 + x_pos), int(size/2 - y_pos)), 0, (255, 255, 255), int(size/4))

    for row in range(0, size):
        print(f'{row} / {size}')
        for col in range(0, size):
            x = (col - x_pos - size / 2.0) * CHI
            y = (size/2.0 - row - y_pos) * CHI
            r = sqrt(x*x + y*y)
            theta = arctan2(y, x)

            x_, y_ = spherical(r, theta, X, Y, GAMMA, CHI)

            row_ = round(size/2.0 - y_ - y_pos)
            col_ = round(x_pos + size/2.0 + x_)

            if 0 <= row_ < size and 0 <= col_ < size:
                dist[row, col] = img[row_, col_]

    img = np.resize(img, (size, size, 1)).astype('uint8')
    dist = np.resize(dist, (size, size, 1)).astype('uint8')

    img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    dist = cv.cvtColor(dist, cv.COLOR_GRAY2BGR)
    img = cv.circle(img, (int(size/2), int(size/2)), 0, (0, 0, 255), 4)
    dist = cv.circle(dist, (int(size/2), int(size/2)), 0, (0, 0, 255), 4)
    img = cv.resize(img, (400, 400))
    dist = cv.resize(dist, (400, 400))
    cv.imshow("img", img)
    cv.imshow("dist", dist)
    cv.waitKey(0)
