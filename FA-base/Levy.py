from math import gamma, sin, pi, ceil
import numpy as np


def levy(n_dim):
    beta = 3 / 2
    sigma = (gamma(1 + beta) * sin(pi * beta / 2) /
             (gamma((1 + beta) / 2) * beta * (2 ** ((beta - 1) / 2)))
             ) ** (1 / beta)
    u = np.random.randn(n_dim) * sigma
    v = np.random.randn(n_dim)
    step = (u / np.power(np.abs(v), 1 / beta))*sign()
    return step

def sign():
    if np.random.rand() < 1/2:
        return -1
    else:
        return 1






