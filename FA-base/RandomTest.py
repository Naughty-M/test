import time

import numpy as np
import matplotlib.pyplot as plt
from FA import FA
import K_means as mean
np.zeros(10)
# print(np.zeros(10),)

if __name__ == '__main__':
    t = np.zeros(10)
    value = np.zeros(10)
    for i in range(10):
        fa = FA(10, 40, 1, 0.000001, 0.97, 500, [-100, 100],3)
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





