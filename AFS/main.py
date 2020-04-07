import random
import time
from functools import reduce

import numpy as np


def func(x_):
    D = 30
    '''x_new1 = x_ ** 2
    return -20 * np.exp(-0.2 * np.sqrt((1 / D) * reduce(lambda x, y: x + y, x_new1))) - \
           np.exp((1 / D) * reduce(lambda x, y: x + y, np.cos(2 * np.pi * x_))) + 20 + np.e'''
    '''return np.linalg.norm(x_) ** 2''' #F2
    '''return np.linalg.norm(x_, ord=1) + abs(np.prod(x_))'''
    # return np.linalg.norm(x_, ord=np.Inf)   #F4
    # result = 0.
    # for n in range(D - 1):
    #     result += 100 * (x_[n + 1] - x_[n] ** 2) ** 2 + (x_[n] - 1) ** 2
    # return result】
    result = 0.
    for n in range(D):
        result += (n + 1) * x_[n] ** 4

    return 100000 + random.random()
    '''result =0.
    for n in range(D):
        result += np.abs(x_[n] + 0.5) ** 2

    return result'''


from sko.AFSA import AFSA
if __name__=="__main__":


    afsa = AFSA(func, n_dim=30, size_pop=30, max_iter=50,
                max_try_num=100, step=0.5, visual=0.3,
                q=0.98, delta=0.5)
    best_x, best_y = afsa.run()
    # print("Xasxasddasd")
    print(best_x, best_y,"dadas da ad adasd asd a")

    # T = 10
    # t = np.zeros(T)
    # value = np.zeros(T)
    # for i in range(T):  ## 问题维数 群体大小 最大吸引度 光吸收系数 步长因子 最大代数  bound
    #     afsa = AFSA(func, n_dim=30, size_pop=30, max_iter=100,
    #                 max_try_num=100, step=0.5, visual=0.3,
    #                 q=0.98, delta=0.5)
    #     time_start = time.time()
    #     best_x, best_y = afsa.run()
    #
    #     time_end = time.time()
    #     t[i] = time_end - time_start
    #     value[i]= best_y
    #     print(value[i])
    # print("平均值：", np.average(value))
    # print("最优值：", np.min(value))
    # print("最差值：", np.max(value))
    # print("平均时间：", np.average(t))

