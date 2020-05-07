import time

import numpy as np
from FA import FA
from Petri_test import MSE

# np.set_printoptions(suppress=True)

if __name__ == '__main__':
    T = 1
    D = 5
    ttime = np.zeros(T)
    value = np.zeros(T)
    MSEvalue = np.zeros(T)
    juti_value = np.zeros((T,D))

    for i in range(T):  ## 问题维数 群体大小 最大吸引度 光吸收系数 步长因子 最大代数  bound
        fa = FA(D, 50, 1, 1.0, 0.97, 200, [0.1, 1],2)
        time_start = time.time()
        ploy_f = fa.copy_iterate()
        time_end = time.time()

        ttime[i] = time_end - time_start
        print(ttime[i],"时间时间时间时间时间时间时间时间时间时间时间时间时间时间时间")
        value[i], n = fa.find_min()
        w = [0.2, 0.5, 0.3, 0.4, 0.6]
        u = [0.7, 0.9, 0.6, 0.8, 0.7]
        t = [0.3, 0.4, 0.2, 0.5, 0.4]
        print(value[i],"适应度")
        print(n,"具体值")
        juti_value[i] = n
        print("MsEas",MSE(w[:D],n))
        MSEvalue[i] = MSE(w[:D],n)
        # print("dsds")

print("平均值：", np.average(value))
print("最优值：", np.min(value))
print("最差值：", np.max(value))
print("平均时间：", np.average(ttime))

print("MSE 最优",np.min(MSEvalue))
print("MSE",MSEvalue)
no=np.argsort(MSEvalue)

print("最优",juti_value[no])


        #k-means
        # centroids, clusterAssment=mean.KMeans(fa.X,3,fa.FitnessValue)
        # # list =
        # mean.showCluster(fa.X,3,centroids,clusterAssment)

#K-means





