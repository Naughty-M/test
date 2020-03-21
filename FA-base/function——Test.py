
import math

import numpy as np


def fan(T,t):
    resule = 1/(1+math.exp((math.log(3)+math.log(99))*t/T-math.log(99)))
    return resule

def qirta(D):
    return  (2*np.sqrt(np.random.rand(D))-1)*np.random.rand(D)/np.random.rand(D)

if __name__ == '__main__':
    T = 10
    print(1.2*math.exp(-600))
    for t in range(T):


        print(qirta(10))