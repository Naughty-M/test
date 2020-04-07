from functools import reduce

import numpy as np
import matplotlib.pyplot as plt
import copy
import time


class FA_orgain:
    def __init__(self, D, N, Beta0, gama, alpha, T, bound):
        self.D = D  # 问题维数
        self.N = N  # 群体大小
        self.Beta0 = Beta0  # 最大吸引度
        self.gama = gama  # 光吸收系数
        self.alpha = alpha  # 步长因子
        self.T = T
        self.X = (bound[1] - bound[0]) * np.random.random([N, D]) + bound[0]
        self.X_origin = copy.deepcopy(self.X)
        self.FitnessValue = np.zeros(N)
        for n in range(N):
            self.FitnessValue[n] = self.FitnessFunction(self.X[n, :])

    def DistanceBetweenIJ(self, i, j):
        return np.linalg.norm(self.X[i, :] - self.X[j, :])

    def BetaIJ(self, i, j):  # AttractionBetweenIJ
        return self.Beta0 * \
               np.math.exp(-self.gama * (self.DistanceBetweenIJ(i, j) ** 2))

    def update(self, i, j):
        self.X[i, :] = self.X[i, :] + \
                       self.BetaIJ(i, j) * (self.X[j, :] - self.X[i, :]) + \
                       self.alpha * (np.random.rand(self.D) - 0.5)

    def FitnessFunction(self, x_):

        # x_ = self.X[i, :]  # X[1,:]是取第1维中下标为1的元素的所有数据，第1行（从0开始）
        return np.linalg.norm(x_)**2     #np.linalg.norm(求范数)   **乘方
        # return np.linalg.norm(x_, ord=1) + abs(np.prod(x_))   #F2   搞不得
        # return np.linalg.norm(x_,ord=np.Inf)
        # return (x_[1]-5.1/(4*(math.pi**2))*x_[0]**2+5/math.pi*x_[0]-6)**2+10*(1-1/(8*math.pi))*math.cos(x_[0])+10   #
        # x_new = (np.abs(x_ + 0.5)) ** 2
        # return reduce(lambda x, y: x + y, x_new)

    def FindNewBest(self, i):
        FFi = self.FitnessFunction(self.X[i, :])
        x_ = self.X[i, :] + self.alpha * (np.random.rand(self.D) - 0.5)
        ffi = self.FitnessFunction(x_)
        if ffi < FFi:
            self.X[i, :] = x_
            self.FitnessValue[i] = ffi

    def iterate(self):
        t = 0
        plot_Fa = []
        while t < self.T:
            for i in range(self.N):
                tag = 0
                FFi = self.FitnessValue[i]
                for j in range(self.N):
                    FFj = self.FitnessValue[j]
                    if FFj < FFi:
                        tag = 1
                        self.update(i, j)
                        self.FitnessValue[i] = self.FitnessFunction(self.X[i, :])
                        FFi = self.FitnessValue[i]
                if tag == 0:
                    self.FindNewBest(i)
            plot_Fa.append(np.min(self.FitnessValue))

            t += 1
        return  plot_Fa

    def find_min(self):
        v = np.min(self.FitnessValue)
        n = np.argmin(self.FitnessValue)
        return v, self.X[n, :]


if __name__ == '__main__':
    t = np.zeros(10)
    value = np.zeros(10)
    for i in range(10):
        fa = FA(30, 30, 1, 1.0, 0.5, 500, [-100, 100])
        time_start = time.time()
        fa.iterate()
        time_end = time.time()
        t[i] = time_end - time_start
        value[i], n = fa.find_min()
    print("平均值：", np.average(value))
    print("最优值：", np.min(value))
    print("最差值：", np.max(value))
    print("平均时间：", np.average(t))