import math

import numpy as np


def GrieFunc(vardim, x_, bound):
    D = vardim
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
    ''' A = np.zeros((2, 25))
    a = [-32, -16, 0, 16, 32]
    A[0, :] = np.tile(a, (1, 5))
    A[1, :] = np.repeat(a, 5)

    result = 0.
    for j in range(25):
        zx1 = (x_[0] - A[0, j]) ** 6 + (x_[1] - A[1, j]) ** 6 + j + 1
        result += 1 / zx1
    # print((0.002 + result) ** (-1),"ahhaahah")
    return (0.002 + result) ** (-1)'''

    # for i in range(1, vardim + 1):
    #
    #     s1 = s1 + x_[i - 1] ** 2
    #     s2 = s2 * math.cos(x[i - 1] / math.sqrt(i))
    # y = (1. / 4000.) * s1 - s2 + 1
    # y = 1. / (1. + y)

    # X[1,:]是取第1维中下标为1的元素的所有数据，第1行（从0开始）
    # return np.linalg.norm(x_) ** 2  # np.linalg.norm(求范数)   **乘方
    # return np.linalg.norm(x_, ord=1) + abs(np.prod(x_))   #F2   搞不得
    # return np.linalg.norm(x_, ord=np.Inf)    #F4
    '''result = 0.
    for n in range(self.D-1):
        result += 100*(x_[n+1]-x_[n]**2)**2+(x_[n]-1)**2
    return result'''  # F5

    '''result =0.
    for n  in range(self.D):
        result+= np.abs(x_[n]+0.5)**2

    return result#F6'''  # F6
    '''result = 0.
    for n in range(self.D):
        result += (n+1)*x_[n]**4

    return  result+random.random()'''  # F7
    '''sqrt_x = np.sqrt(np.abs(x_))
    x_new = x_*np.sin(sqrt_x)
    # print(sqrt_x,"x_new")
    # print(418.9828*self.D)
    print(reduce(lambda x, y: x + y, x_new),"reduce ")
    return 418.9828*self.D - reduce(lambda x, y: x + y, x_new)# F8=cannot'''

    result = 0.
    for n in range(D):

        result += x_[n]**2 - 10*np.cos(2*np.pi*x_[n])+10
    return result  # F9

    '''x_new1 = x_**2
    return -20*np.exp(-0.2*np.sqrt((1/self.D)*reduce(lambda x, y: x + y,x_new1)))-\
           np.exp((1/self.D)*reduce(lambda x, y: x + y,np.cos(2*np.pi*x_)))+20+np.e'''  # F10
    '''x_new1 = x_ ** 2
    result = 1
    for n in range(self.D):
         result*=np.cos(x_[n]/np.sqrt(n+1))

    return 1/4000*reduce(lambda x, y: x + y,x_new1)-result+1'''

    '''A = np.zeros((2, 25))
    a = [-32, -16, 0, 16, 32]
    A[0, :] = np.tile(a, (1, 5))
    A[1, :] = np.repeat(a, 5)
    result = 0.
    for j in range(25):
        zx1 = (x_[0]-A[0,j])**6+(x_[1]-A[1,j])**6+j+1
        result+=1/zx1
    return  (0.002+result)**(-1)'''  # F14




def RastFunc(vardim, x, bound):
    """
    Rastrigin function
    """
    # s = 10 * 25
    for i in range(1, vardim + 1):
        s = s + x[i - 1] ** 2 - 10 * math.cos(2 * math.pi * x[i - 1])


    return 100