import numpy as np
import matplotlib.pyplot as plt
import copy
import time
from functools import singledispatch   #重载函数
import random
import Levy as Levy


class FA:
    def __init__(self, D, N, Beta0, gama, alpha, T, bound,mean):
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
        self.mean = mean  #聚类长度
        self.bound  = bound

        self.X = (bound[1] - bound[0]) * np.random.random([N, D]) + bound[0]    #我觉得是初始化

        self.X_origin = copy.deepcopy(self.X)
        """
        copy function is used to  copy list   
        """
        self.FitnessValue = np.zeros(N)  # fitness value  返回的是一个个 浮点0【0.,0.,0.,0,.0,.0,.0,.0.】数组
        for n in range(N):
            self.FitnessValue[n] = self.FitnessFunction(n)

    def adjust_alphat(self, t,i,j):
        if self.DistanceBetweenIJ(i,j)<self.alpha:
            self.alpha = 0.4/(1+np.math.exp(0.015*(t-self.T)/3)) # 自适应步长

    def DistanceBetweenIJ(self, i, j):
        # if i==j:
        #     return 0.
        return np.linalg.norm(self.X[i, :] - self.X[j, :])   #求范数    距离  OK

    def BetaIJ(self, i, j):  # AttractionBetweenIJ
        return self.Beta0 * \
               np.math.exp(-self.gama * (self.DistanceBetweenIJ(i, j) ** 2))    #吸引度

    def update(self, i, j):
        self.X[i, :] = self.X[i, :] + \
                       self.BetaIJ(i, j) * (self.X[j, :] - self.X[i, :]) + \
                       self.alpha * (np.random.rand(self.D) - 0.5)   #np.random.rand(self.D)对应维度的数组

    def I_average_Distance(self,i):         #计算精英萤火虫i的平均距离   list 为
        """
        :param N:  邻居大小
        :param i: 传入的萤火虫
        :return: 平均距离
        """
        N = self.mean
        list = np.argsort(self.FitnessValue)
        list = list[:N]
        sum_distance = 0.
        for j in list:
            if(i!=j):
                # print(self.DistanceBetweenIJ(i,j))
                sum_distance+=self.DistanceBetweenIJ(i,j)
        i_avg_distance = sum_distance/ N-1
        return  i_avg_distance

    def retrun_neighbour(self,i): #返回邻居
        list = np.argsort(self.FitnessValue)
        for j in list[:self.mean]:
            if(self.DistanceBetweenIJ(i,j) <= self.I_average_Distance(i)*random.random() and j!=i):
                return j
        return -1

    def compare_ijFitness(self,i,j):    #比较 i和j 的适应度大小    i>j  retrue True
            if(self.FitnessValue[i]>self.FitnessValue[j]):
                return True
            else:
                return False

    def update_neighboru(self,i,t):
        """
        :param i:
        :param j:
        """
        j= self.retrun_neighbour(i)
        if(j!=-1):   #判断是否为i的邻居
            if(self.compare_ijFitness(i,j)):
                self.adjust_alphat(t,i,j)
                self.X[i,:] = self.X[i, :] + \
                              self.BetaIJ(i, j)*np.random.rand(self.D)*(self.X[j,:]-self.X[i,:])+ \
                              np.linalg.norm(self.X[j,:]-self.X[i,:])*self.alpha/(self.bound[1]-self.bound[0])
                # print("有比i强的邻居")

            else:
                self.X[i,:] = self.bound[1]-self.bound[0]-self.X[i,:]
                # print("是邻居 但强度不够")
                # self.X[i, :] = self.X[i, :] + Levy.levy(self.D)
        else:
            # print("邻居都没有")
            self.X[i,:] = self.X[i,:]+Levy.levy(self.D)\
                          # *(self.bound[1]-self.bound[0])

    def copy_iterate(self):
        t = 0
        while t < self.T:  # 迭代代数
            Kmeanslist = self.KMeans()
            for i in range(self.N):
                sort_list = np.argsort(self.FitnessValue)
                # print(self.X[list])
                if i in sort_list[:self.mean]:
                    self.update_neighboru(i,t)
                else:

                    for j in range(self.N) :
                        if i == j:
                            continue
                          #kmeans 列表
                        if(Kmeanslist[j]==Kmeanslist[i]):
                            # print("Kmeanslist[j]==Kmeanslist[i]",i)
                            FFi = self.FitnessValue[i]
                            FFj = self.FitnessValue[j]
                            if FFj < FFi:
                                self.adjust_alphat(t, i, j)  # 自适应步长
                                self.update(i, j)
                                self.FitnessValue[i] = self.FitnessFunction(i)
                                # FFi = self.FitnessValue[i]

            # self.K_mean_Plot()
            # Fly_plot(self.X)
            t += 1
            print("tttttttttt",t)

    def FitnessFunction(self, i):
        x_ = self.X[i, :]            #X[1,:]是取第1维中下标为1的元素的所有数据，第1行（从0开始）
        return np.linalg.norm(x_) ** 2     #np.linalg.norm(求范数)   **乘方

    def iterate(self):  #迭代     move
        t = 0
        sort_list = np.argsort(self.FitnessValue)
        while t < self.T:     #迭代代数
            for i in range(self.N):
                FFi = self.FitnessValue[i]

                for j in range(self.N):
                    self.adjust_alphat(t,i,j)  #自适应步长
                    FFj = self.FitnessValue[j]
                    if FFj < FFi:
                        # self.update_neighboru(i,j)
                        self.update(i, j)
                        self.FitnessValue[i] = self.FitnessFunction(i)
                        FFi = self.FitnessValue[i]
            # self.K_mean_Plot()
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

    def K_mean_Plot(self):  #我自都不知道是啥了   哦哦哦 画图
        centroids, clusterAssment = self.KMeans(self.X, self.mean, self.FitnessValue)
        self.showCluster(self.X, self.mean, centroids, clusterAssment)

    def showCluster(dataSet, k, centroids, clusterAssment):
        m, n = dataSet.shape
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
            plt.plot(dataSet[i, 0], dataSet[i, 1], mark[markIndex])
        mark = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']
        # 绘制质心
        for i in range(k):
            plt.plot(centroids[i, 0], centroids[i, 1], mark[i])
        # plt.pause(0.5)
        # plt.clf()

    def set_Cent(self):

        sort_list = np.argsort(self.FitnessValue)
        centroids =  np.zeros((self.mean, self.D))

        for i in range(self.mean):
            index = sort_list[i]  #
            centroids[i, :] = self.X[index, :]
        return centroids  #    centroids 为[质心的编号  质心的坐标  ]

    def KMeans(self):
        m = self.N  # 行的数目 种群大小
        # 第一列存样本属于哪一簇
        # 第二列存样本的到簇的中心点的误差
        # clusterAssment = np.mat(np.zeros((m, 2)))
        # clusterChange = True
        sort_List_fitness = np.argsort(self.FitnessValue)    #适应度列表
        k_Means_list = np.zeros(self.N)#聚类列表
        # 第1步 初始化centroids
        # centroids = randCent(dataSet, k)
        # centroids = self.set_Cent()  # 中心聚类   centroids 为[质心的编号  质心的坐标  ]
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
                print(j,"K-mean")
                distance = self.DistanceBetweenIJ(i,j)
                if distance < minDist:
                    minDist = distance
                    minIndex = j
            # 第 3 步：更新每一行样本所属的簇
            # if clusterAssment[i, 0] != minIndex:
                # clusterChange = True
                # clusterAssment[i, :] = minIndex, minDist ** 2   # 第一列存样本属于哪一簇    # 第二列存样本的到簇的中心点的误差   暂时用不到
            k_Means_list[i]=minIndex
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

    def K_means_list_I(self):  #返回I的簇内元素
        list_keams_I =self.KMeans()
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

def Fly_plot(X):    #萤火虫轨迹
    plt.xlabel('x')
    plt.ylabel('y')
    plt.scatter(X[:, 0], X[:, 1], c='g')
    #
    plt.pause(0.5)
    plt.clf()



if __name__ == '__main__':
    t = np.zeros(10)
    value = np.zeros(10)        ## 问题维数 群体大小 最大吸引度 光吸收系数 步长因子 最大代数  bound
    for i in range(10):
        fa = FA(2, 20, 1, 0.000001, 0.97, 50, [-100, 100],3)
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