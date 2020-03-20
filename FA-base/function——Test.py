import time

import numpy as np
import random as rd
import matplotlib.pyplot as plt
import copy

from math import *


class DE:
    def __init__(self, size, dim, maxgen, bound, param):
        self.size = size
        self.maxgen = maxgen
        self.dim = dim
        self.bound = bound
        self.param = param
        self.p = np.zeros((size, dim))
        self.m = np.zeros((size, dim))  # 变异坐标
        self.c = np.zeros((size, dim))  # 交叉坐标
        self.bp = [0.0 for i in range(dim)]  # bp为最优解的 坐标 1行
        self.f = np.zeros((size, 1))  # f为初始点的适应度函数值 n行1列  数组
        self.trace = []  # trace为轨迹 记录 每次迭代的最佳适应度函数 列表类型

    def begin(self):
        for i in range(0, self.size):
            # for j in range(0,self.dim):
            self.p[i] = self.bound[0] + rd.uniform(0, 1) * (self.bound[1] - self.bound[0])
            self.f[i] = self.fitness(self.p[i])
        # print(self.p)
        # print(self.f)

    def fitness(self, x_):
        # 用的是病态函数

        # f=-(math.fabs(math.sin(x[0])*math.cos(x[1])*math.exp(math.fabs(1-math.sqrt(x[0]**2+x[1]**2)/math.pi))))
        # f=-(fabs(sin(x[0])*cos(x[1])*exp(fabs(1-sqrt(x[0]**2+x[1]**2)/pi))))
        # f=-0.0001*(fabs(sin(x[0])*sin(x[1])*exp(fabs(100-sqrt(x[0]**2+x[1]**2)/pi))+1))**0.1
        return np.linalg.norm(x_) ** 2
        # return np.linalg.norm(x_, ord=1) + np.prod(list(map(abs, x_)))

    def bianyi(self, t):
        for i in range(t):
            '''z=exp(1-(self.maxgen/(self.maxgen+1-t)))
            self.param[0]=0.25*2**z'''
            self.param[0] = 0.8 - t * (0.8 - 0.4) / self.maxgen
        for i in range(self.size):
            r1 = r2 = r3 = 0
            while r1 == i or r2 == i or r3 == i or r2 == r1 or r3 == r1 or r3 == r2:
                r1 = rd.randint(0, self.size - 1)  # 随机数范围为[0,size-1]的整数
                r2 = rd.randint(0, self.size - 1)
                r3 = rd.randint(0, self.size - 1)
            self.m[i] = self.p[r1] + (self.p[r2] - self.p[r3]) * self.param[0]
            for j in range(self.dim):
                if self.m[i, j] < self.bound[0]:
                    self.m[i, j] = self.bound[0]
                if self.m[i, j] > self.bound[1]:
                    self.m[i, j] = self.bound[1]

    def crossover(self):

        for i in range(self.size):
            for j in range(self.dim):
                if rd.uniform(0, 1) <= self.param[1] or j == rd.randint(0, self.size - 1):
                    self.c[i, j] = self.m[i, j]
                else:
                    self.c[i, j] = self.p[i, j]

    def selection(self):

        for i in range(self.size):
            if self.fitness(self.c[i]) <= self.fitness(self.p[i]):
                self.bp = self.c[i]  #
                self.p[i] = self.c[i]
            else:

                self.bp = self.bp
                self.p[i] = self.p[i]

    def run(self):
        self.begin()
        # print((self.p))
        bestindex = np.where(self.f == np.min(self.f))

        # print('最佳索引为：',bestindex)
        self.bp = self.p[bestindex]  # 初始化后默认的最佳一行的位置
        '''for i in range(self.size):
            if self.fitness(self.p[i])<=self.fitness(self.bp):

                self.bp=self.p[i]
        print((self.bp))'''

        for t in range(self.maxgen):
            self.bianyi(t)
            self.crossover()
            self.selection()
            if self.fitness(self.bp) < 1e-2:
                print('达到最佳的迭代次数:', t)

            '''if self.fitness(self.bp)+19.2085<1e-4:
                print('达到最佳的迭代次数:',t)'''

            self.trace.append(self.fitness(self.bp))
            # print(self.trace)

            # print(self.flist)
            # print(self.fitness(self.mm))

        return self.trace


def main():
    T = 10
    t = np.zeros(T)
    value = np.zeros(T)
    for i in range(10):
        de = DE(size=100, dim=30, maxgen=2000, bound=[-100, 100], param=[0.8, 0.6])
        time_start = time.time()
        de.run()

        #
        print('=' * 40)
        print('= Optimal solution:')
        print('=   x=', de.bp[0])
        print('=   y=', de.bp[1])
        print('= Function value:')
        value[i] = de.fitness(de.bp)
        print('=   f(x,y)=', de.fitness(de.bp))
        # print(np.shape(de.bp))

        # list,list2=fa.KMeans()
        # print(list2,"************")
        # print(fa.K_means_list_I())
        time_end = time.time()
        t[i] = time_end - time_start
        print('=' * 40)
        # print(de.flist)
        # print(de.bp)

        #
        # plt.plot(de.trace, 'r')
        # title = 'MIN: ' + str(de.fitness(de.bp))
        # plt.title(title)
        # plt.xlabel("Number of iterations")
        # plt.ylabel("Function values")
        # plt.show()

    print("平均值：", np.average(value))
    print("最优值：", np.min(value))
    print("最差值：", np.max(value))
    print("平均时间：", np.average(t))

if __name__ == "__main__":
    main()


