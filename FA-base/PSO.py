import copy
from functools import reduce

import numpy as np
import random

import matplotlib.pyplot as plt

from Petri_test import sigmoid, Funvtion


def fit_fun(x_,D):  # 适应函数
    return Funvtion(x_)

    """
    result = 0.
    xx= np.copy(x_)
    for i in range(D):
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

    return result
    """


    # x_ = np.array(x_)
    # X[1,:]是取第1维中下标为1的元素的所有数据，第1行（从0开始）
    # return np.linalg.norm(x_) ** 2  # np.linalg.norm(求范数)   **乘方
    # return np.linalg.norm(x_, ord=1) + abs(np.prod(x_))   #F2   搞不得
    # return np.linalg.norm(x_, ord=np.Inf)    #F4

    # return 0;
    '''result = 0.
    for n in range(D-1):
        result += 100*(x_[n+1]-x_[n]**2)**2+(x_[n]-1)**2
    return result  # F5'''

    # result =0.
    '''for n  in range(D):
        result+= np.abs(x_[n]+0.5)**2

    return result#F6 # F6'''
    ''' result = 0.
    for n in range(D):
        result += (n+1)*x_[n]**4

    return  result+random.random() # F7'''
    '''sqrt_x = np.sqrt(np.abs(x_))
    x_new = x_*np.sin(sqrt_x)
    # print(sqrt_x,"x_new")
    # print(418.9828*self.D)
    print(reduce(lambda x, y: x + y, x_new),"reduce ")
    return 418.9828*self.D - reduce(lambda x, y: x + y, x_new)# F8=cannot'''

    '''result = 0.
    for n in range(D):

        result += x_[n]**2 - 10*np.cos(2*np.pi*x_[n])+10
    return result # F9'''
    # print(x_)

    '''x_new1 = x_** 2
    return -20 * np.exp(-0.2 * np.sqrt((1 / D) * reduce(lambda x, y: x + y, x_new1))) - \
           np.exp((1 / D) * reduce(lambda x, y: x + y, np.cos(2 * np.pi * x_))) + 20 + np.e  # F10'''
    # x_new1 = x_ ** 2
    # result = 1
    # for n in range(D):
    #      result*=np.cos(x_[n]/np.sqrt(n+1))
    #
    # return 1/4000*reduce(lambda x, y: x + y,x_new1)-result+1#F11


    '''A = np.zeros((2, 25))
    a = [-32, -16, 0, 16, 32]
    A[0, :] = np.tile(a, (1, 5))
    A[1, :] = np.repeat(a, 5)
    result = 0.
    for j in range(25):
        zx1 = (x_[0]-A[0,j])**6+(x_[1]-A[1,j])**6+j+1
        result+=1/zx1
    return  (0.002+result)**(-1)# F14'''

    # return -np.abs(np.sin(X[0]) * np.cos(X[1]) * np.exp(np.abs(1 - np.sqrt(X[0] ** 2 + X[1] ** 2) / np.pi)))
    # return -np.abs(np.sin(X[0]) * np.cos(X[1]) * np.exp(np.abs(1 - np.sqrt(X[0] ** 2 + X[1] ** 2) / np.pi)))


class Particle:
    # 初始化
    def __init__(self, x_max, max_vel, dim):
        self.__pos = [random.uniform(0, x_max) for i in range(dim)]  # 粒子的位置
        self.__vel = [random.uniform(-max_vel, max_vel) for i in range(dim)]  # 粒子的速度
        self.__bestPos = [0.0 for i in range(dim)]  # 粒子最好的位置
        self.__fitnessValue = fit_fun(self.__pos,dim)  # 适应度函数值

    def set_pos(self, i, value):
        self.__pos[i] = value

    def get_pos(self):
        return self.__pos

    def set_best_pos(self, i, value):
        self.__bestPos[i] = value

    def get_best_pos(self):
        return self.__bestPos

    def set_vel(self, i, value):
        self.__vel[i] = value

    def get_vel(self):
        return self.__vel

    def set_fitness_value(self, value):
        self.__fitnessValue = value

    def get_fitness_value(self):
        return self.__fitnessValue


class PSO:
    def  __init__(self, dim, size, iter_num, x_max, max_vel, best_fitness_value=float('Inf'), C1=2, C2=2, W=1):
        self.C1 = C1
        self.C2 = C2
        self.W = W
        self.dim = dim  # 粒子的维度
        self.size = size  # 粒子个数
        self.iter_num = iter_num  # 迭代次数
        self.x_max = x_max
        self.max_vel = max_vel  # 粒子最大速度
        self.best_fitness_value = best_fitness_value
        self.best_position = [0.0 for i in range(dim)]  # 种群最优位置
        self.fitness_val_list = []  # 每次迭代最优适应值

        # 对种群进行初始化
        self.Particle_list = [Particle(self.x_max, self.max_vel, self.dim) for i in range(self.size)]

    def set_bestFitnessValue(self, value):
        self.best_fitness_value = value

    def get_bestFitnessValue(self):
        return self.best_fitness_value

    def set_bestPosition(self, i, value):
        self.best_position[i] = value

    def get_bestPosition(self):
        return self.best_position

    # 更新速度
    def update_vel(self, part):
        for i in range(self.dim):
            vel_value = self.W * part.get_vel()[i] + self.C1 * random.random() * (part.get_best_pos()[i] - part.get_pos()[i]) \
                        + self.C2 * random.random() * (self.get_bestPosition()[i] - part.get_pos()[i])
            if vel_value > self.max_vel:
                vel_value = self.max_vel
            elif vel_value < -self.max_vel:
                vel_value = -self.max_vel
            part.set_vel(i, vel_value)

    # 更新位置
    def update_pos(self, part):
        for i in range(self.dim):
            pos_value = part.get_pos()[i] + part.get_vel()[i]
            part.set_pos(i, pos_value)
        value = fit_fun(part.get_pos(),self.dim)
        if value < part.get_fitness_value():
            part.set_fitness_value(value)
            for i in range(self.dim):
                part.set_best_pos(i, part.get_pos()[i])
        if value < self.get_bestFitnessValue():
            self.set_bestFitnessValue(value)
            for i in range(self.dim):
                self.set_bestPosition(i, part.get_pos()[i])

    def update(self):
        list_x = []
        for i in range(self.iter_num):
            for part in self.Particle_list:
                self.update_vel(part)  # 更新速度
                self.update_pos(part)  # 更新位置
            self.fitness_val_list.append(self.get_bestFitnessValue())  # 每次迭代完把当前的最优适应度存到列表


            list_x.append(copy.deepcopy(self.get_bestPosition()))
            # print("grtthe best position",self.get_bestPosition())
            print("PSO---i==",i)
        # print(list_x)
        print(list_x)
        return self.fitness_val_list,list_x
        # return self.fitness_val_list, self.get_bestPosition()


def function_PSO(dim, size, iter_num, x_max, max_vel):
    # dim = 5
    # size = 50
    # iter_num = 200
    # x_max = 1
    # max_vel = 0.5  # 粒子最大的速度
    pso = PSO(dim, size, iter_num, x_max, max_vel)
    fit_var_list1, position_list = pso.update()
    return fit_var_list1,position_list

if __name__ == "__main__":
    T = 1
    t = np.zeros(T)
    value = np.zeros(T)
    for i in range(T):  ## 问题维数 群体大小 最大吸引度 光吸收系数 步长因子 最大代数
        PSO_y,PSO_position = function_PSO(5,50,200,1,0.5)
        # print("PSO最优位置:" + str(best_pos1))
        value[i] = PSO_y[-1]
        print(i,"i",value[i])
        # print("PSO最优解:" + str(fit_var_list1[-1]))
    print("平均值：", np.average(value))
    print("最优值：", np.min(value))
    print("最差值：", np.max(value))
    # plt.plot(np.linspace(0, iter_num, iter_num), fit_var_list1, c="R", alpha=0.5, label="PSO")
    # plt.show()