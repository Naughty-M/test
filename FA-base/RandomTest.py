import numpy as np
import matplotlib.pyplot as plt
from FA import FA
import K_means as mean
np.zeros(10)
# print(np.zeros(10),)

if __name__ == '__main__':
    t = np.zeros(10)
    value = np.zeros(10)  ## 问题维数 群体大小 最大吸引度 光吸收系数 步长因子 最大代数  bound
    for i in range(1):
        fa = FA(2, 200, 1, 0.000001, 0.97, 50, [-100, 100])
        # fa.show_data()
        # print(fa.np_sort())
        # fa.iterate()
        centroids, clusterAssment=mean.KMeans(fa.X,3)
        mean.showCluster(fa.X,3,centroids,clusterAssment)

#K-means


    # dataSet = 0
    # k = 4
    # centroids, clusterAssment = KMeans(dataSet, k)


