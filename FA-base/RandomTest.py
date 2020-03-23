import time

import numpy as np
import matplotlib.pyplot as plt
from FA import FA
import K_means as mean
np.zeros(10)
# print(np.zeros(10),)

if __name__ == '__main__':
    T = 30
    t = np.zeros(T)
    value = np.zeros(T)
    for i in range(T):  ## 问题维数 群体大小 最大吸引度 光吸收系数 步长因子 最大代数  bound
        fa = FA(30, 30, 1, 1.0, 0.5, 500, [-100, 100],3)
        time_start = time.time()
        fa.copy_iterate()
        # list,list2=fa.KMeans()
        # print(list2,"************")
        # print(fa.K_means_list_I())
        time_end = time.time()
        t[i] = time_end - time_start
        value[i], n = fa.find_min()
        print(value[i])
print("平均值：", np.average(value))
print("最优值：", np.min(value))
print("最差值：", np.max(value))
print("平均时间：", np.average(t))


        #k-means
        # centroids, clusterAssment=mean.KMeans(fa.X,3,fa.FitnessValue)
        # # list =
        # mean.showCluster(fa.X,3,centroids,clusterAssment)

#K-means





