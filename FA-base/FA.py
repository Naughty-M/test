import math

import numpy as np
import matplotlib.pyplot as plt
import copy
import time
from functools import singledispatch, reduce  # 重载函数
import random
import Levy as Levy
from Petri_test import sigmoid, Funvtion


# np.set_printoptions(suppress=True)


class FA:
    def __init__(self, D, N, Beta0, gama, alpha, T, bound, mean):
        '''

        :param D:问题维度
        :param N:群体大小
        :param Beta0:最大吸引度
        :param gama:光吸收系数
        :param alpha: 步长因子
        :param T: best generation
        :param bound:  I think is the bound
        :param mean :聚类长度
        '''

        self.D = D  # 问题维数
        self.N = N  # 群体大小
        self.Beta0 = Beta0  # 最大吸引度
        self.gama = gama  # 光吸收系数
        self.alpha = alpha  # 步长因子
        self.T = T
        self.mean = mean  # 聚类长度
        self.bound = bound
        self.alpha2 = self.alpha  # tjjx
        self.X = self.initial()
        self.X_origin = copy.deepcopy(self.X)
        """
        copy function is used to  copy list   
        """

        self.FitnessValue = np.zeros(N)  # fitness value  返回的是一个个 浮点0【0.,0.,0.,0,.0,.0,.0,.0.】数组
        self.sortList = self.np_sort()

        for n in range(N):
            self.FitnessValue[n] = self.FitnessFunction(self.X[n, :])

    def initial(self):
        return (self.bound[1] - self.bound[0]) * np.random.random([self.N, self.D]) + self.bound[0]  # 我觉得是初始化

    # def adjust_alphat(self, t, i, j):
    #     if self.DistanceBetweenIJ(i, j) < self.alpha:
    #         self.alpha = 0.4 / (1 + np.math.exp(0.015 * (t - self.T) / 3))  # 自适应步长
    #     else:
    #         self.alpha = 0.95

    def t_adjust_alphat(self, t):  # 只有代数的步长策略
        # self.alpha = self.alpha**2  # 自适应步长
        # self.alpha *= random.random()
        # self.alpha = np.exp(-t / self.T) * self.alpha
        # self.alpha = self.alpha / (1 + np.math.exp(0.015 * (t - self.T) ))*self.alpha # 自适应步长

        a = math.e**(t/self.T)
        self.alpha*= 1.0/(a + a*self.alpha) #论文用


        # self.alpha = (1 - t / self.T) * self.alpha
        # self.alpha = 0.4 / (1 + np.math.exp(0.015 * (t - self.T) / 3))*self.alpha
        #

    def DistanceBetweenIJ(self, i, j):
        # if i==j:
        #     return 0.
        return np.linalg.norm(self.X[i, :] - self.X[j, :])  # 求范数    距离  OK

    def t_adjust_gama(self, t):
        # self.gama = (t/self.T)*self.gama
        self.gama = math.sin(self.gama * math.pi) * t / self.T

    def BetaIJ(self, i, j):  # AttractionBetweenIJ
        return self.Beta0 * \
               np.math.exp(-self.gama * (self.DistanceBetweenIJ(i, j) ** 2))  # 吸引度

    def fan(self, t):  # 振荡
        T = self.T
        resule = 1 / (1 + math.exp((math.log(3) + math.log(99)) * t / T - math.log(99)))
        # print(resule,"resule")    趋向   0.5
        return resule

    def zhengdang(self, t):
        if (t <= self.T / 2):
            r = np.random.rand(self.D)
            # print(r, "**************************************************************************")
            # return (2*np.sqrt(np.random.rand(self.D)+0.00000000001)-1)*(np.random.rand(self.D)+0.00000000001+1)/(r+0.00000000001)
            return (2 * np.sqrt(np.random.rand(self.D) + 0.00000000001) - 1) * (
                    np.random.rand(self.D) + 0.00000000001) / (np.random.rand(self.D) + 0.00000000001)

        else:
            return (2 * np.sqrt(np.random.rand(self.D) + 0.00000000001) - 1) * (
                        np.random.rand(self.D) + 0.00000000001) / (np.random.rand(self.D) + 0.00000000001)

    def update(self, i, j, t):
        # self.fan(t) * ((2*np.sqrt(np.random.rand(self.D))-1)*np.random.rand(self.D)/np.random.rand(self.D)*\
        self.X[i, :] = (self.X[i, :] + self.BetaIJ(i, j) * (self.X[j, :] - self.X[i, :]) +
                        self.alpha * (np.random.rand(self.D) - 0.5))  # np.random.rand(self.D)对应维度的数组

        #

    def I_average_Distance(self, i):  # 计算精英萤火虫i的平均距离   list 为
        """
        :param N:  邻居大小
        :param i: 传入的萤火虫
        :return: 平均距离
        """
        N = self.mean
        list = self.sortList
        list = list[:N]
        sum_distance = 0.
        for j in list:
            if (i != j):
                # print(self.DistanceBetweenIJ(i,j))
                sum_distance += self.DistanceBetweenIJ(i, j)
        i_avg_distance = sum_distance / N - 1
        return i_avg_distance

    def update_neighboru(self, i, t):
        """
        :param i:
        :param j:
        """
        x_ = np.copy(self.X[i, :])
        '''if (i == self.sortList[0]):
            for  n in range(self.D ):
                o = self.X[i, n] + (Levy.levy(1)) * self.alpha
                if(o< self.bound[1]  and o>self.bound[0]):
                    x_[n]=o
            x_value= self.FitnessFunction(x_)
            if x_value  <self.FitnessValue[i]:
                self.X[i,:] = x_
                self.FitnessValue[i] = x_value'''  # 随机
        if (i == self.sortList[0]):
            for n in range(self.D):
                o = self.X[i, n] + (Levy.levy(1)) * self.alpha
                if (o <= self.bound[1] and o >= self.bound[0]):
                    x_[n] = np.copy(o)
                else:
                    if o>self.bound[1]:
                        o += self.bound[1]+x_[n]*(1-random.random())
                        o%=self.bound[1]-self.bound[0]
                    else:
                        o = self.bound[0]+x_[n]*(1-random.random())
                        o %= self.bound[1] - self.bound[0]
                    x_[n] = np.copy(o)

                if (self.FitnessFunction(x_) < self.FitnessFunction(self.X[i,:])):
                    self.X[i, :] = np.copy(x_)
                else:
                    x_ = np.copy(self.X[i, :])

                # print(x_)
                # print(self.X[i])
                # print("************")

            # x_ = self.X[i, :] + (Levy.levy(self.D)) * self.alpha
            self.X[i, :] = np.copy(x_)
        else:
            # k = 0
            for j in self.sortList[0:self.mean]:
                if (self.FitnessValue[i] > self.FitnessValue[j]):  # i>j  True
                    self.X[i, :] = (self.zhengdang(t) * self.X[i, :] +
                                    self.BetaIJ(i, j) * np.random.rand(self.D) * (self.X[j, :] - self.X[i, :]) +
                                    np.linalg.norm(self.X[j, :] - self.X[i, :]) * self.alpha * (np.random.rand(self.D) - 0.5) /
                                    (self.bound[1] - self.bound[0]))

                    # self.X[i,:] = self.zhengdang(t)*self.X[i, :] + \
                    #                  self.BetaIJ(i, j)*np.random.rand(self.D)*(self.X[j,:]-self.X[i,:])+ \
                    #                  np.linalg.norm(self.X[j,:]-self.X[i,:])*self.alpha/(self.bound[1]-self.bound[0]) #精英令居


                    # self.alpha * (Levy.levy(self.D)*np.random.rand(self.D))    #levy  飞行
                    # 添加了(2*np.sqrt(np.random.rand(self.D))-1)*np.random.rand(self.D)/np.random.rand(self.D)   多维度
                    # print("有比i强的邻居")

    def copy_iterate(self):
        t = 0
        plot_Fa = []
        FA_position = []
        while t < self.T:  # 迭代代数

            self.t_adjust_alphat(t)
            self.t_adjust_gama(t)
            self.sortList = self.np_sort()
            sort_list = self.sortList  # is ok
            centroids, clusterAssment = self.Kmeans_Two_parament()
            # self.showCluster(centroids, clusterAssment)
            for i in range(self.N):
                # print("for i in range(self.N)",i)
                # print(self.X[list])
                if i in sort_list[:self.mean]:
                    self.update_neighboru(i, t)
                    self.FitnessValue[i] = self.FitnessFunction(self.X[i, :])
                    # if self.FitnessFunction(self.X[i, :]) ==0:
                    #     print(self.X[i,:],"FitnessFunction")
                else:
                    for j in range(self.N):
                        # kmeans 列表
                        # print(clusterAssment,"clusterAssment")
                        if (clusterAssment[i, 0] == clusterAssment[j, 0]):
                            FFi = self.FitnessValue[i]
                            FFj = self.FitnessValue[j]
                            if FFj < FFi:
                                # self.adjust_alphat(t, i, j)  # 自适应步长
                                self.update(i, j, t)
                                self.FitnessValue[i] = self.FitnessFunction(self.X[i, :])

            t += 1
            print("========FA ,i",t)
            # 自然选择简单
            # select = 0
            # while select < self.mean * (1 - t / self.T):
            #     self.X[self.sortList[self.N - 1 - select], :] = self.X[self.sortList[select], :]
            #     self.FitnessValue[self.sortList[self.N - 1 - select]] = self.FitnessFunction(
            #         self.X[self.sortList[self.N - 1 - select], :])
            #     select += 1

            # self.K_mean_Plot()
            # Fly_plot(self.X)
            # self.showCluster(centroids,clusterAssment)
            # print(t,"ttttttttttttttttttttttttttt")

            # print(t, "代数")
            #
            # print(np.min(self.FitnessValue), "最优值")
            # # print(self.FitnessValue[self.sortList[0]],"self.FitnessValue[sort_list[0]]")
            # print(np.array(self.alpha), "步长")
            # print(self.X[self.sortList[0]])
            plot_Fa.append(self.FitnessValue.min())
            oo = self.X[sort_list[0]]
            FA_position.append(oo.tolist())
            # print(oo.tolist())

            # print("tttttttttt",t)
            # print(FA_position)
        return plot_Fa,FA_position

    def FitnessFunction(self, x_):

        return Funvtion(x_)
        """result = 0.
        xx= np.copy(x_)
        for i in range(self.D):
            w = [0., 0.2, 0.5, 0.3, 0.4, 0.6]
            u = [0., 0.7, 0.9, 0.6, 0.8, 0.7]
            t = [0., 0.3, 0.4, 0.2, 0.5, 0.4]

            p1 = 0.9
            p4 = 0.9
            p5 = 0.9
            p7 = 0.9
            b = 5000
            # t[4] = x_[0]
            ww = copy.deepcopy(w)
            uu = copy.deepcopy(u)
            tt = copy.deepcopy(t)
            if(i<5):
                w = [0., 0.2, 0.5, 0.3, 0.4, 0.6]

                w[i + 1] = xx[i] + 0.0
                p9 = p1 * u[1] * sigmoid(p1, b, t[1])
                p2 = p1 * u[2] * sigmoid(p1, b, t[2])
                x1 = p9
                x2 = p2 * u[3] * sigmoid(p2, b, t[3])
                # p3 = max(x1, x2)
                p3 = x1 * sigmoid(x1, b, x2) + x2 * sigmoid(x1, b, x2)
                x3 = p4 * w[1] + p3 * w[2] + p5 * w[3]
                p6 = x3 * u[4] * sigmoid(x3, b, t[4])
                x4 = p6 * w[4] + p7 * w[5]
                p8 = x4 * u[5] * sigmoid(x4, b, t[5])
                result += (p8 - 0.603792) ** 2
                # result +=(p8 - 0.568015203115994)**2
            elif i<10:
                u[i+1-5] = xx[i]+0.0
                p9 = p1 * u[1] * sigmoid(p1, b, t[1])
                p2 = p1 * u[2] * sigmoid(p1, b, t[2])
                x1 = p9
                x2 = p2 * u[3] * sigmoid(p2, b, t[3])
                # p3 = max(x1, x2)
                p3 = x1 * sigmoid(x1, b, x2) + x2 * sigmoid(x1, b, x2)
                x3 = p4 * w[1] + p3 * w[2] + p5 * w[3]
                p6 = x3 * u[4] * sigmoid(x3, b, t[4])
                x4 = p6 * w[4] + p7 * w[5]
                p8 = x4 * u[5] * sigmoid(x4, b, t[5])
                result += (p8 - 0.568015203115994) ** 2
            else:
                t[i + 1-10] = xx[i] + 0.0
                p9 = p1 * u[1] * sigmoid(p1, b, t[1])
                p2 = p1 * u[2] * sigmoid(p1, b, t[2])
                x1 = p9
                x2 = p2 * u[3] * sigmoid(p2, b, t[3])
                # p3 = max(x1, x2)
                p3 = x1 * sigmoid(x1, b, x2) + x2 * sigmoid(x1, b, x2)
                x3 = p4 * w[1] + p3 * w[2] + p5 * w[3]
                p6 = x3 * u[4] * sigmoid(x3, b, t[4])
                x4 = p6 * w[4] + p7 * w[5]
                p8 = x4 * u[5] * sigmoid(x4, b, t[5])
                # result += (p8 - 0.568015203115994) ** 2
                # result += (p8 - 0.603792) ** 2
                # pass

        return result"""

        # X[1,:]是取第1维中下标为1的元素的所有数据，第1行（从0开始）
        # return np.linalg.norm(x_)**2     #np.linalg.norm(求范数)   **乘方
        # return np.linalg.norm(x_, ord=1) + abs(np.prod(x_))   #F2   搞不得
        # return np.linalg.norm(x_, ord=np.Inf)    #F4
        '''result = 0.
        for n in range(self.D-1):
            result += 100*(x_[n+1]-x_[n]**2)**2+(x_[n]-1)**2
        return result  #F5'''

        '''result =0.
        for n  in range(self.D):
            result+= np.abs(x_[n]+0.5)**2

        return result#F6  #F6'''
        '''result = 0.
        for n in range(self.D):
            result += (n+1)*x_[n]**4

        return  result+random.random()  #F7'''
        '''sqrt_x = np.sqrt(np.abs(x_))
        x_new = x_*np.sin(sqrt_x)
        # print(sqrt_x,"x_new")
        # print(418.9828*self.D)
        print(reduce(lambda x, y: x + y, x_new),"reduce ")
        return 418.9828*self.D - reduce(lambda x, y: x + y, x_new)# F8=cannot'''

        '''result = 0.
        for n in range(self.D):

            result += x_[n]**2 - 10*np.cos(2*np.pi*x_[n])+10
        return result  #F9'''

        '''x_new1 = x_**2
        return -20*np.exp(-0.2*np.sqrt((1/self.D)*reduce(lambda x, y: x + y,x_new1)))-\
               np.exp((1/self.D)*reduce(lambda x, y: x + y,np.cos(2*np.pi*x_)))+20+np.e #F10'''
        ''' x_new1 = x_ ** 2
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
        return  (0.002+result)**(-1)    '''  # F14

        # return np.linalg.norm(x_,ord=np.Inf)
        # return (x_[1]-5.1/(4*(math.pi**2))*x_[0]**2+5/math.pi*x_[0]-6)**2+10*(1-1/(8*math.pi))*math.cos(x_[0])+10   #

        # x_new = (np.abs(x_ + 0.5)) ** 2
        #
        # return reduce(lambda x, y: x + y, x_new)+

        """x_new = (-1)*x_ * np.sin(np.sqrt(abs(x_)))
        return reduce(lambda x, y: x + y, x_new)"""

        '''x_new = x_*np.sin(10*np.pi*x_)
        return (-1)*reduce(lambda x, y: x + y, x_new)'''

    def iterate(self):  # 迭代     move
        t = 0
        # sort_list = self.sortList
        while t < self.T:  # 迭代代数
            print(np.min(self.FitnessValue))
            self.t_adjust_alphat(t)
            # self.t_adjust_gama(t)
            for i in range(self.N):
                FFi = self.FitnessValue[i]

                for j in range(self.N):
                    FFj = self.FitnessValue[j]
                    if FFj < FFi:
                        # self.adjust_alphat(t,i,j)  #自适应步长
                        self.update(i, j, t)
                        self.FitnessValue[i] = self.FitnessFunction(self.X[i, :])
                        FFi = self.FitnessValue[i]
            # Fly_plot(self.X)
            t += 1
            print(t)
            # print(np.min(self.FitnessValue))

    def find_min(self):
        v = np.min(self.FitnessValue)
        n = np.argmin(self.FitnessValue)  # 返回最小索引
        return v, self.X[n, :]

    def np_sort(self):
        return np.argsort(self.FitnessValue)

    def show_data(self):
        i = 0
        while i < self.N:
            print(i)
            print(self.FitnessValue[i])
            print(self.X[i, :])
            print("****")
            i += 1

    # 欧氏距离计算
    def distEclud(self, x, y):
        return np.sqrt(np.sum((x - y) ** 2))  # 计算欧氏距离

    def K_mean_Plot(self):  # 我自都不知道是啥了   哦哦哦 画图

        centroids, clusterAssment, list = self.KMeans(self.X, self.mean, self.FitnessValue)
        self.showCluster(self.X, self.mean, centroids, clusterAssment)

    def showCluster(self, centroids, clusterAssment):
        # plt.figure(figsize=(self.bound[0],self[1]))
        # plt.axis(self.bound[0],self.bound[1],self.bound[0],self.bound[1])
        # plt.xlim(self.bound[0],self.bound[1])
        # plt.ylim(self.bound[0], self.bound[1])
        k = self.mean

        m, n = self.X.shape
        if n != 2:
            print("数据不是二维的")
            return 1
        mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']
        if k > len(mark):
            print("k值太大了")
            return 1
        # 绘制所有的样本
        for i in range(m):
            markIndex = int(clusterAssment[i, 0])
            plt.plot(self.X[i, 0], self.X[i, 1], mark[markIndex])
        mark = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']
        # 绘制质心
        for i in range(k):
            plt.plot(centroids[i, 0], centroids[i, 1], mark[i])

        plt.show()
        # plt.pause(0.5)
        # plt.clf()

    def set_Cent(self):
        sort_list = self.sortList
        centroids = np.zeros((self.mean, self.D))

        for i in range(self.mean):
            index = sort_list[i]  #
            centroids[i, :] = self.X[index, :]
        return centroids  # centroids 为[质心的编号  质心的坐标  ]

    def Kmeans_Two_parament(self):
        m = self.N  # 行的数目
        # 第一列存样本属于哪一簇
        # 第二列存样本的到簇的中心点的误差
        clusterAssment = np.mat(np.zeros((m, 2)))
        clusterChange = True

        # 第1步 初始化centroids
        # centroids = randCent(dataSet, k)

        centroids = self.set_Cent()  # 中心聚类   centroids 为[ 质心的坐标  ]
        while clusterChange:
            clusterChange = False
            # 遍历所有的样本（行数）
            for i in range(m):
                minDist = 100000000000.0
                minIndex = 0

                # 遍历所有的质心
                # 第2步 找出最近的质心
                for j in range(self.mean):
                    # 计算该样本到质心的欧式距离
                    # print(centroids[j, :],"centroids[j, :]")
                    distance = self.distEclud(centroids[j, :], self.X[i, :])
                    if distance <= minDist:
                        minDist = distance
                        minIndex = j
                # 第 3 步：更新每一行样本所属的簇
                if clusterAssment[i, 0] != minIndex:
                    clusterChange = True
                    # print(minIndex,"minIndex")
                    clusterAssment[i, :] = minIndex, minDist ** 2
                if (minIndex == -1):
                    pass
                    # print(distance,"distance")
        return centroids, clusterAssment

    def KMeans(self):
        m = self.N  # 行的数目 种群大小
        # 第一列存样本属于哪一簇
        # 第二列存样本的到簇的中心点的误差
        clusterAssment = np.mat(np.zeros((m, 2)))
        # clusterChange = True
        sort_List_fitness = self.sortList  # 适应度列表
        k_Means_list = np.zeros(self.N)  # 聚类列表
        # 第1步 初始化centroids
        # centroids = randCent(dataSet, k)
        centroids = self.set_Cent()  # 中心聚类   centroids 为[质心的编号  质心的坐标  ]
        # while clusterChange:
        #     clusterChange = False
        # 遍历所有的样本（行数）
        for i in range(m):
            minDist = 100000.0
            minIndex = -1
            # 遍历所有的质心
            # 第2步 找出最近的质心
            for j in sort_List_fitness[:self.mean]:
                # 计算该样本到质心的欧式距离
                # print(j,"K-mean")
                distance = self.DistanceBetweenIJ(i, j)
                if distance < minDist:
                    minDist = distance
                    minIndex = j
            # 第 3 步：更新每一行样本所属的簇
            # if clusterAssment[i, 0] != minIndex:
            # clusterChange = True
            # clusterAssment[i, :] = minIndex, minDist ** 2   # 第一列存样本属于哪一簇    # 第二列存样本的到簇的中心点的误差   暂时用不到
            k_Means_list[i] = minIndex
            # else:
            #     k_Means_list[i] = sort_List_fitness[i]
            # 第 4 步：更新质心  我取消了
            # for j in range(k):
            #     pointsInCluster = dataSet[np.nonzero(clusterAssment[:, 0].A == j)[0]]  # 获取簇类所有的点
            #     centroids[j, :] = np.mean(pointsInCluster, axis=0)  # 对矩阵的行求均值
        # print("Congratulations,cluster complete!")
        # print(centroids)  # 质心集合

        # print(clusterAssment)

        # print("************")
        # return centroids, clusterAssment,k_Means_list
        return k_Means_list

    def K_means_list_I(self):  # 返回I的簇内元素
        list_keams_I = self.KMeans()
        # print(list_keams_I)
        return list_keams_I


def plot(X_origin, X):
    fig_origin = plt.figure(0)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.scatter(X_origin[:, 0], X_origin[:, 1], c='r')
    plt.scatter(X[:, 0], X[:, 1], c='g')
    plt.pause(0.1)
    plt.clf()


def Fly_plot(X):  # 萤火虫轨迹
    plt.xlabel('x')
    plt.ylabel('y')
    plt.scatter(X[:, 0], X[:, 1], c='g')
    #
    plt.pause(0.001)
    plt.clf()


if __name__ == '__main__':
    t = np.zeros(10)
    value = np.zeros(10)  ## 问题维数 群体大小 最大吸引度 光吸收系数 步长因子 最大代数  bound
    for i in range(10):
        fa = FA(5, 50, 1, 1.0, 0.5, 500, [0, 1], 3)
        # print(fa.FitnessValue)
        # fa.np_sort()
        # print(fa.FitnessValue)
        time_start = time.time()
        fa.iterate()
        time_end = time.time()
        t[i] = time_end - time_start
        value[i], n = fa.find_min()
        print(value[i], "最优值")
        print(t, "迭代次数")
        # plot(fa.X_origin, fa.X)
    print("平均值：", np.average(value))
    print("最优值：", np.min(value))
    print("最差值：", np.max(value))
    print("平均时间：", np.average(t))
