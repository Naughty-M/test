import math

import numpy as np


def GrieFunc(vardim, x_, bound):
    """
    Griewangk function
    """
    ''' s1 = 0.
     s2 = 1.
     for i in range(1, vardim + 1):
         s1 = s1 + x[i - 1] ** 2
         s2 = s2 * math.cos(x[i - 1] / math.sqrt(i))
     y = (1. / 4000.) * s1 - s2 + 1
     y = 1. / (1. + y)
     # print(abs(y)+10
     return y'''
    '''result = 0.
    for n in range(vardim):
        result += x_[n]**2 - 10*np.cos(2*np.pi*x_[n])+10
    return result'''   #F9
    A = np.zeros((2, 25))
    a = [-32, -16, 0, 16, 32]
    A[0, :] = np.tile(a, (1, 5))
    A[1, :] = np.repeat(a, 5)

    result = 0.
    for j in range(25):
        zx1 = (x_[0] - A[0, j]) ** 6 + (x_[1] - A[1, j]) ** 6 + j + 1
        result += 1 / zx1
    # print((0.002 + result) ** (-1),"ahhaahah")
    return (0.002 + result) ** (-1)




def RastFunc(vardim, x, bound):
    """
    Rastrigin function
    """
    s = 10 * 25
    for i in range(1, vardim + 1):
        s = s + x[i - 1] ** 2 - 10 * math.cos(2 * math.pi * x[i - 1])


    return s**2