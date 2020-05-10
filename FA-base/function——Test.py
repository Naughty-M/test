
import math
from functools import reduce

import numpy as np

import matplotlib.pyplot as plt
def fan(T,t):
    resule = 1/(1+math.exp((math.log(3)+math.log(99))*t/T-math.log(99)))
    return resule

def qirta(D):
    return  (2*np.sqrt(np.random.rand(D))-1)*np.random.rand(D)/np.random.rand(D)
def FitnessFunction( x_,D):
    print(x_,"x_")
    sqrt_x = np.sqrt(np.abs(x_))
    print(sqrt_x,"sqrt_x")
    x_new = np.sin(sqrt_x)
    print(x_new,"x_nre")
    x_new = x_*np.sin(sqrt_x)
    print(x_new,"x_new")
    return 418.9828 * D - reduce(lambda x, y: x + y, x_new)  # F8=cannot

if __name__ == '__main__':
    list= [0.14540498390483736, 0.5704281973893393, 0.376368173632254, 0.37443397100987813, 0.6027774925633882]
    print(list[3::-1])