import numpy as np
import matplotlib.pyplot as plt
import copy
import time
from functools import singledispatch   #重载函数
import K_means as mean

class FA:
    def __init__(self, D, N, Beta0, gama, alpha, T, bound,mean):
        '''

        :param D:群体大小
        :param N:群体大小
        :param Beta0:最大吸引度
        :param gama:光吸收系数
        :param alpha: 步长因子
        :param T: best generation
        :param bound:  I think is the bound
        '''

        self.D = D  # 问题维数
        self.N = N  # 群体大小
        self.Beta0 = Beta0  # 最大吸引度
        self.gama = gama  # 光吸收系数
        self.alpha = alpha  # 步长因子
        self.T = T
        self.mean = mean  #聚类长度

        self.X = (bound[1] - bound[0]) * np.random.random([N, D]) + bound[0]    #我觉得是初始化

        self.X_origin = copy.deepcopy(self.X)
        """
        copy function is used to  copy list   
        """
        self.FitnessValue = np.zeros(N)  # fitness value  返回的是一个个 浮点0【0.,0.,0.,0,.0,.0,.0,.0.】数组
        for n in range(N):
            self.FitnessValue[n] = self.FitnessFunction(n)

    def adjust_alphat(self, t):
        self.alpha = (1 - t / self.T) * self.alpha # 自适应步长



    def DistanceBetweenIJ(self, i, j):
        return np.linalg.norm(self.X[i, :] - self.X[j, :])   #求范数    距离  OK

    def BetaIJ(self, i, j):  # AttractionBetweenIJ
        return self.Beta0 * \
               np.math.exp(-self.gama * (self.DistanceBetweenIJ(i, j) ** 2))    #吸引度

    def update(self, i, j):

        self.X[i, :] = self.X[i, :] + \
                       self.BetaIJ(i, j) * (self.X[j, :] - self.X[i, :]) + \
                       self.alpha * (np.random.rand(self.D) - 0.5)

    def FitnessFunction(self, i):
        x_ = self.X[i, :]            #X[1,:]是取第1维中下标为1的元素的所有数据，第1行（从0开始）
        return np.linalg.norm(x_) ** 2     #np.linalg.norm(求范数)   **乘方

    def iterate(self):  #迭代     move
        t = 0
        while t < self.T:     #迭代代数
            self.adjust_alphat(t)   #自适应步长

            for i in range(self.N):
                FFi = self.FitnessValue[i]
                for j in range(self.N):
                    FFj = self.FitnessValue[j]
                    if FFj < FFi:
                        self.update(i, j)
                        self.FitnessValue[i] = self.FitnessFunction(i)
                        FFi = self.FitnessValue[i]

            self.K_mean_Plot()
            # Fly_plot(self.X)
            t += 1



    def find_min(self):
        v = np.min(self.FitnessValue)
        n = np.argmin(self.FitnessValue)      #返回最小索引
        return v, self.X[n, :]

    def np_sort(self):

        return np.argsort(-self.FitnessValue)



    def show_data(self):
        i = 0
        while i<self.N:
            print(i)
            print(self.FitnessValue[i])
            print(self.X[i,:])
            print("****")
            i+=1

    def K_mean_Plot(self):
        centroids, clusterAssment = mean.KMeans(self.X, self.mean, self.FitnessValue)
        # mean.showCluster(self.X, self.mean, centroids, clusterAssment)


def plot(X_origin, X):
    fig_origin = plt.figure(0)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.scatter(X_origin[:, 0], X_origin[:, 1], c='r')
    plt.scatter(X[:, 0], X[:, 1], c='g')
    plt.pause(0.1)
    plt.clf()

def Fly_plot(X):    #萤火虫轨迹
    plt.xlabel('x')
    plt.ylabel('y')
    plt.scatter(X[:, 0], X[:, 1], c='g')
    #
    # plt.pause(0.005)
    # plt.clf()





if __name__ == '__main__':
    t = np.zeros(10)
    value = np.zeros(10)        ## 问题维数 群体大小 最大吸引度 光吸收系数 步长因子 最大代数  bound
    for i in range(10):
        fa = FA(2, 20, 1, 0.000001, 0.97, 50, [-100, 100])
        # print(fa.FitnessValue)
        # fa.np_sort()
        # print(fa.FitnessValue)



        time_start = time.time()
        fa.iterate()
        time_end = time.time()
        t[i] = time_end - time_start
        value[i], n = fa.find_min()
        plot(fa.X_origin, fa.X)
    print("平均值：", np.average(value))
    print("最优值：", np.min(value))
    print("最差值：", np.max(value))
    print("平均时间：", np.average(t))