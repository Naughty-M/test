from functools import reduce

import numpy as np
import random

import matplotlib.pyplot as plt
def fit_fun(x_):  # 适应函数
    D = 30
    x_ = np.array(x_)
    # X[1,:]是取第1维中下标为1的元素的所有数据，第1行（从0开始）
    # return np.linalg.norm(x_) ** 2  # np.linalg.norm(求范数)   **乘方
    # return np.linalg.norm(x_, ord=1) + abs(np.prod(x_))   #F2   搞不得
    # return np.linalg.norm(x_, ord=np.Inf)    #F4
    '''result = 0.
    for n in range(self.D-1):
        result += 100*(x_[n+1]-x_[n]**2)**2+(x_[n]-1)**2
    return result'''  # F5

    '''result =0.
    for n  in range(self.D):
        result+= np.abs(x_[n]+0.5)**2

    return result#F6'''  # F6
    '''result = 0.
    for n in range(self.D):
        result += (n+1)*x_[n]**4

    return  result+random.random()'''  # F7
    '''sqrt_x = np.sqrt(np.abs(x_))
    x_new = x_*np.sin(sqrt_x)
    # print(sqrt_x,"x_new")
    # print(418.9828*self.D)
    print(reduce(lambda x, y: x + y, x_new),"reduce ")
    return 418.9828*self.D - reduce(lambda x, y: x + y, x_new)# F8=cannot'''

    '''result = 0.
    for n in range(self.D):

        result += x_[n]**2 - 10*np.cos(2*np.pi*x_[n])+10
    return result'''  # F9
    print(x_)
    x_new1 = x_** 2
    return -20 * np.exp(-0.2 * np.sqrt((1 / D) * reduce(lambda x, y: x + y, x_new1))) - \
           np.exp((1 / D) * reduce(lambda x, y: x + y, np.cos(2 * np.pi * x_))) + 20 + np.e  # F10
    '''x_new1 = x_ ** 2
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
    return  (0.002+result)**(-1)'''  # F14

    # return -np.abs(np.sin(X[0]) * np.cos(X[1]) * np.exp(np.abs(1 - np.sqrt(X[0] ** 2 + X[1] ** 2) / np.pi)))
    # return -np.abs(np.sin(X[0]) * np.cos(X[1]) * np.exp(np.abs(1 - np.sqrt(X[0] ** 2 + X[1] ** 2) / np.pi)))


class Particle:
    # 初始化
    def __init__(self, x_max, max_vel, dim):
        self.__pos = [random.uniform(-x_max, x_max) for i in range(dim)]  # 粒子的位置
        self.__vel = [random.uniform(-max_vel, max_vel) for i in range(dim)]  # 粒子的速度
        self.__bestPos = [0.0 for i in range(dim)]  # 粒子最好的位置
        self.__fitnessValue = fit_fun(self.__pos)  # 适应度函数值

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
    def __init__(self, dim, size, iter_num, x_max, max_vel, best_fitness_value=float('Inf'), C1=2, C2=2, W=1):
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
        value = fit_fun(part.get_pos())
        if value < part.get_fitness_value():
            part.set_fitness_value(value)
            for i in range(self.dim):
                part.set_best_pos(i, part.get_pos()[i])
        if value < self.get_bestFitnessValue():
            self.set_bestFitnessValue(value)
            for i in range(self.dim):
                self.set_bestPosition(i, part.get_pos()[i])

    def update(self):
        for i in range(self.iter_num):
            for part in self.Particle_list:
                self.update_vel(part)  # 更新速度
                self.update_pos(part)  # 更新位置
            self.fitness_val_list.append(self.get_bestFitnessValue())  # 每次迭代完把当前的最优适应度存到列表
        return self.fitness_val_list, self.get_bestPosition()


if __name__ == "__main__":
    dim = 30
    size = 30
    iter_num = 500
    x_max = 100
    max_vel = 0.05   #粒子最大的速度


    pso = PSO(dim, size, iter_num, x_max, max_vel)
    fit_var_list1, best_pos1 = pso.update()
    print("PSO最优位置:" + str(best_pos1))
    print("PSO最优解:" + str(fit_var_list1[-1]))
    # plt.plot(np.linspace(0, iter_num, iter_num), fit_var_list1, c="R", alpha=0.5, label="PSO")
    # plt.show()