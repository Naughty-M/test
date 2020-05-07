# coding: utf-8

# In[88]:

import numpy as np
from numpy import random as rd
import matplotlib.pyplot as plt
import pandas as pd
import time

np.set_printoptions(linewidth=1000, suppress=True)

# In[89]:
# 从TSPLIB官网上下载TSP数据kroA100
with open("./kroA100.tsp/data", "r", encoding="utf-8") as file:
    data = file.read()
data = np.array([i.split(" ")[1:] for i in data.split("\n")[6:-2]]).astype(np.float)


# In[90]:

def GetTime(func_name):
    def inner(*args, **kwargs):
        start_time = time.time()
        ret = func_name(*args, **kwargs)
        end_time = time.time()
        run_time = end_time - start_time
        print("运行时间%f秒" % run_time)
        return ret

    return inner


# In[91]:

class MMAS(object):

    def __init__(self, city_position, initial_pheromone, max_pheromone, min_pheromone, run_times, ant_count, alpha,
                 beta, Q, w, tau, q_0):
        """
            city_position:城市坐标，N行2列，N表示城市个数，2列分别为x坐标以及y坐标
            initial_pheromone:初始信息素值
            max_pheromone:信息素最大值
            min_pheromone:信息素最小值
            run_times:算法跑几代
            ant_count:蚂蚁数量
            alpha:选择城市时信息素重要性
            beta:选择城市时启发信息重要性
            Q:信息素更新时用到的常数
            w:信息素的挥发百分比, 0到1之间
            tau:信息素平滑系数，0到1之间
            q_0:每只蚂蚁在状态转移时采取随机型还是确定型，0到1之间。随机产生一个0到1之间的随机数，
                如果随机数大于q_0，则可能选择被选概率最大的城市；如果小于q_0，则一定选择被选概率最大的城市
        """
        self.city_position = city_position
        self.max_pheromone = max_pheromone
        self.min_pheromone = min_pheromone
        self.run_times = run_times
        self.ant_count = ant_count
        self.city_count = self.city_position.shape[0]
        self.distance_matrix, self.heuristic_information = self.get_distance_matrix()
        self.pheromone_matrix = np.ones_like(self.distance_matrix) * initial_pheromone - np.diag(
            [initial_pheromone] * self.city_count, k=0)
        self.alpha = alpha
        self.beta = beta
        self.current_best_path = None
        self.current_best_distance = float("inf")
        self.Q = Q
        self.w = w
        self.tau = tau
        self.best_distance_lst = []
        self.q_0 = q_0

    def get_distance_matrix(self):
        """
            根据城市坐标获取城市距离矩阵，启发信息矩阵
        """
        for i in range(self.city_position.shape[0]):
            distance = np.sqrt(np.sum(np.power(self.city_position[i] - self.city_position, 2), axis=1, keepdims=True))
            if not i:
                result = distance
                continue
            result = np.concatenate([result, distance], axis=1)
        return result, pd.DataFrame(1 / result).replace(float("inf"), 0).values

    def get_proba(self, not_passed_city, passed_city):
        """
            获取所有未经过城市被选取的概率
        """
        not_passed_city = np.array(not_passed_city)
        current_city_of_every_ant = np.array(passed_city)[:, -1]
        pheromone_next_path = []
        heuristic_next_path = []
        for i in range(self.ant_count):
            pheromone_next_path.append(
                self.pheromone_matrix[[current_city_of_every_ant[i]] * not_passed_city.shape[1], not_passed_city[i]])
            heuristic_next_path.append(self.heuristic_information[
                                           [current_city_of_every_ant[i]] * not_passed_city.shape[1], not_passed_city[
                                               i]])
        pheromone_heuristic = np.power(pheromone_next_path, self.alpha) * np.power(heuristic_next_path, self.beta)
        prob = pheromone_heuristic / np.sum(pheromone_heuristic, axis=1, keepdims=True)
        return prob

    def get_path_distance(self, passed_city):
        """
            遍历完所有城市后，获取每只蚂蚁行走路径长度
        """
        distance_of_every_ant = [0] * self.ant_count
        for i in range(self.ant_count):
            for j in range(self.city_count - 1):
                start_city = passed_city[i][j]
                end_city = passed_city[i][j + 1]
                distance_of_every_ant[i] += self.distance_matrix[start_city, end_city]
            distance_of_every_ant[i] += self.distance_matrix[passed_city[i][-1], passed_city[i][0]]
        return np.array(distance_of_every_ant)

    def get_pheromone_update_matrix(self, current_generation_best_path, current_generation_best_distance):
        """
            获取信息素更新矩阵
        """
        pheromone_update_matrix = np.zeros_like(self.pheromone_matrix)
        if self.current_best_path == current_generation_best_path:
            # 当前代最优路径和当前最优路径为同一条路径
            current_update_value = self.Q / self.current_best_distance
            for i in range(self.city_count - 1):
                start = self.current_best_path[i]
                end = self.current_best_path[i + 1]
                pheromone_update_matrix[start, end] += current_update_value
            pheromone_update_matrix[self.current_best_path[-1], self.current_best_path[0]] += current_update_value
        else:
            # 当前代最优路径和当前最优路径不是同一条路径
            current_generation_update_value = self.Q / current_generation_best_distance
            current_update_value = self.Q / self.current_best_distance
            for i in range(self.city_count - 1):
                current_start = self.current_best_path[i]
                current_generation_start = current_generation_best_path[i]
                current_end = self.current_best_path[i + 1]
                current_generation_end = current_generation_best_path[i + 1]
                pheromone_update_matrix[current_start, current_end] += current_update_value
                pheromone_update_matrix[
                    current_generation_start, current_generation_end] += current_generation_update_value
            pheromone_update_matrix[self.current_best_path[-1], self.current_best_path[0]] += current_update_value
            pheromone_update_matrix[
                current_generation_best_path[-1], current_generation_best_path[0]] += current_generation_update_value
        return pheromone_update_matrix

    def update_pheromone_matrix(self, pheromone_update_matrix):
        """
            信息素矩阵更新
        """
        self.pheromone_matrix = (1 - self.w) * self.pheromone_matrix + pheromone_update_matrix
        bigger_than_max_index = self.pheromone_matrix > self.max_pheromone
        smaller_than_min_index = self.pheromone_matrix < self.min_pheromone
        self.pheromone_matrix[bigger_than_max_index] = self.max_pheromone
        self.pheromone_matrix[smaller_than_min_index] = self.min_pheromone
        # 信息素平滑
        self.pheromone_matrix += (self.tau * (self.max_pheromone - self.pheromone_matrix))
        self.pheromone_matrix = self.pheromone_matrix - np.diag(
            [self.pheromone_matrix[0, 0]] * self.pheromone_matrix.shape[0])

    @GetTime
    def run(self):
        """
            返回最佳路径以及路径长度
        """
        for i in range(1, self.run_times + 1):
            # 构造蚂蚁已经过城市列表passed_city，以及未经过城市列表not_passed_city
            passed_city = [[rd.randint(0, self.city_count)] for i in range(self.ant_count)]
            not_passed_city = [list(set(range(self.city_count)) - set(i)) for i in passed_city]
            # 当存在未遍历的城市就执行循环体,直到所有的城市遍历完跳出循环
            while np.unique(not_passed_city).shape[0]:
                # 选择下一个城市
                select_prob = self.get_proba(not_passed_city, passed_city)
                q = rd.random()
                if q > self.q_0:
                    # 随机型
                    cum_select_prob = np.cumsum(select_prob, axis=1)
                    select_city_index = []
                    for i in range(self.ant_count):
                        rand_num = rd.random()
                        select_city_index.append(list(rand_num < cum_select_prob[i]).index(True))
                else:
                    # 确定型
                    select_city_index = np.argmax(select_prob, axis=1)
                for i in range(self.ant_count):
                    passed_city[i].append(not_passed_city[i].pop(select_city_index[i]))
            # 混合方式更新信息素
            distance_of_every_ant = self.get_path_distance(passed_city)
            # 找出当前代最短路径及其长度
            best_index = np.argmin(distance_of_every_ant)
            current_generation_best_path = passed_city[best_index]
            current_generation_best_distance = distance_of_every_ant[best_index]
            # 更新当前最优路径及其长度
            if current_generation_best_distance < self.current_best_distance:
                self.current_best_distance = current_generation_best_distance
                self.current_best_path = current_generation_best_path
            pheromone_update_matrix = self.get_pheromone_update_matrix(current_generation_best_path,
                                                                       current_generation_best_distance)
            # 更新信息素矩阵
            self.update_pheromone_matrix(pheromone_update_matrix)
            self.best_distance_lst.append(self.current_best_distance)
        return self.current_best_path, self.current_best_distance


# In[92]:

mmas = MMAS(city_position=data, initial_pheromone=1000, max_pheromone=100, min_pheromone=50, run_times=2000,
            ant_count=50, alpha=1, beta=2, Q=10000, w=0.1, tau=0.5, q_0=0.8)

# In[93]:

mmas.run()

# In[94]:

mmas.best_distance_lst

# In[ ]:


