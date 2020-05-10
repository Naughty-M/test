# -*- coding: cp936 -*-
import copy
import time
from functools import reduce

import numpy as np
import matplotlib.pyplot as plt
import math
import random


# Rastrigr 函数
from Petri_test import sigmoid, Funvtion


def object_function(x_):


    return Funvtion(x_)
    '''
    
    :param x_: 
    :return: 
    '''
    """
    result = 0.
    D= 5

    for i in range(D):
        w = [0., 0.2, 0.5, 0.3, 0.4, 0.6]
        u = [0., 0.7, 0.9, 0.6, 0.8, 0.7]
        t = [0., 0.3, 0.4, 0.2, 0.5, 0.4]
        w[i + 1] = x_[i]
        # t[4] = x_[0]
        ww = copy.deepcopy(w)
        uu = copy.deepcopy(u)
        tt = copy.deepcopy(t)
        p1 = 0.9
        p4 = 0.9
        p5 = 0.9
        p7 = 0.9
        b = 10
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
        # print("result",result)
        # print(i,"i")
        # print("x",x_)
        # print(x_)

    return result
    """




    # D = 30
    # X[1,:]是取第1维中下标为1的元素的所有数据，第1行（从0开始）
    # return np.linalg.norm(x_) ** 2  # np.linalg.norm(求范数)   **乘方
    # return np.linalg.norm(x_, ord=1) + abs(np.prod(x_))   #F2   搞不得
    # return np.linalg.norm(x_, ord=np.Inf)    #F4
    '''result = 0.
    for n in range(D-1):
        result += 100*(x_[n+1]-x_[n]**2)**2+(x_[n]-1)**2
    return result  # F5'''

    '''result =0.
    for n  in range(D):
        result+= np.abs(x_[n]+0.5)**2

    return result#F6  # F6'''
    '''result = 0.
    for n in range(D):
        result += (n+1)*x_[n]**4

    return  result+random.random() # F7'''
    '''sqrt_x = np.sqrt(np.abs(x_))
    x_new = x_*np.sin(sqrt_x)
    # print(sqrt_x,"x_new")
    # print(418.9828*self.D)
    # print(reduce(lambda x, y: x + y, x_new),"reduce ")
    return 418.9828*D - reduce(lambda x, y: x + y, x_new    )# F8=cannot'''

    '''result = 0.
    for n in range(D):

        result += x_[n]**2 - 10*np.cos(2*np.pi*x_[n])+10
    return result  # F9'''

    """x_ = np.array(x_)
    x_new1 = x_**2
    return -20*np.exp(-0.2*np.sqrt((1/D)*reduce(lambda x, y: x + y,x_new1)))-\
           np.exp((1/D)*reduce(lambda x, y: x + y,np.cos(2*np.pi*x_)))+20+np.e # F10"""
    # x_  = np.array(x_)
    # x_new1 = x_ ** 2
    # result = 1
    # for n in range(D ):
    #      result*=np.cos(x_[n]/np.sqrt(n+1))
    #
    # return 1/4000*reduce(lambda x, y: x + y,x_new1)-result+1

    ''' A = np.zeros((2, 25))
    a = [-32, -16, 0, 16, 32]
    A[0, :] = np.tile(a, (1, 5))
    A[1, :] = np.repeat(a, 5)
    result = 0.
    for j in range(25):
        zx1 = (x_[0]-A[0,j])**6+(x_[1]-A[1,j])**6+j+1
        result+=1/zx1
    return  (0.002+result)**(-1)  # F14'''

    # return np.linalg.norm(x_,ord=np.Inf)
    # return (x_[1]-5.1/(4*(math.pi**2))*x_[0]**2+5/math.pi*x_[0]-6)**2+10*(1-1/(8*math.pi))*math.cos(x_[0])+10   #

    # x_new = (np.abs(x_ + 0.5)) ** 2
    #
    # return reduce(lambda x, y: x + y, x_new)+

    """x_new = (-1)*x_ * np.sin(np.sqrt(abs(x_)))
    return reduce(lambda x, y: x + y, x_new)"""

    '''x_new = x_*np.sin(10*np.pi*x_)
    return (-1)*reduce(lambda x, y: x + y, x_new)'''


# 参数
def initpara():
    NP = 50  # 种群数量
    F = 0.6  # 缩放因子
    CR = 0.7  # 交叉概率
    generation = 200  # 遗传代数
    len_x = 5    #维度
    value_up_range = 1
    value_down_range = -value_up_range
    return NP, F, CR, generation, len_x, value_up_range, value_down_range


# 种群初始化
def initialtion(NP,len_x,value_down_range,value_up_range):
    np_list = []  # 种群，染色体
    for i in range(0, NP):
        x_list = []  # 个体，基因
        for j in range(0, len_x):
            x_list.append(value_down_range + random.random() * (value_up_range - value_down_range))
        np_list.append(x_list)
    return np_list


# 列表相减
def substract(a_list, b_list):
    a = len(a_list)
    new_list = []
    for i in range(0, a):
        new_list.append(a_list[i] - b_list[i])
    return new_list


# 列表相加
def add(a_list, b_list):
    a = len(a_list)
    new_list = []
    for i in range(0, a):
        new_list.append(a_list[i] + b_list[i])
    return new_list


# 列表的数乘
def multiply(a, b_list):
    b = len(b_list)
    new_list = []
    for i in range(0, b):
        new_list.append(a * b_list[i])
    return new_list


# 变异
def mutation(np_list,NP,F):
    v_list = []
    for i in range(0, NP):
        r1 = random.randint(0, NP - 1)
        while r1 == i:
            r1 = random.randint(0, NP - 1)
        r2 = random.randint(0, NP - 1)
        while r2 == r1 | r2 == i:
            r2 = random.randint(0, NP - 1)
        r3 = random.randint(0, NP - 1)
        while r3 == r2 | r3 == r1 | r3 == i:
            r3 = random.randint(0, NP - 1)

        v_list.append(add(np_list[r1], multiply(F, substract(np_list[r2], np_list[r3]))))
    return v_list


# 交叉
def crossover(np_list, v_list,NP,len_x,CR):
    u_list = []
    for i in range(0, NP):
        vv_list = []
        for j in range(0, len_x):
            if (random.random() <= CR) | (j == random.randint(0, len_x - 1)):
                vv_list.append(v_list[i][j])
            else:
                vv_list.append(np_list[i][j])
        u_list.append(vv_list)
    return u_list


# 选择
def selection(u_list, np_list,NP):
    for i in range(0, NP):
        if object_function(u_list[i]) <= object_function(np_list[i]):
            np_list[i] = u_list[i]
        else:
            np_list[i] = np_list[i]
    return np_list


def DE(NP, F, CR, generation, len_x, value_up_range, value_down_range):
    # 主函数
    # NP, F, CR, generation, len_x, value_up_range, value_down_range = initpara()
    np_list = initialtion(NP,len_x,value_down_range,value_up_range)
    min_x = []
    min_f = []

    # np_list = []  # 种群，染色体
    for i in range(0, NP):
        xx = []
        xx.append(object_function(np_list[i]))
    min_f.append(min(xx))
    min_x.append(np_list[xx.index(min(xx))])
    for i in range(0, generation):
        v_list = mutation(np_list,NP,F)
        u_list = crossover(np_list, v_list,NP,len_x,CR)
        np_list = selection(u_list, np_list,NP)
        for i in range(0, NP):
            xx = []
            xx.append(object_function(np_list[i]))
        min_f.append(min(xx))
        min_x.append(np_list[xx.index(min(xx))])
        print("DE---i==",i)
    # 输出

    min_ff = min(min_f)  #最小适应度值
    min_xx = min_x[min_f.index(min_ff)]
    print('the minimum point is x ')
    print(min_xx)
    print('the minimum value is y ')
    print(min_ff)

    # min_f 最小适应度
    # min_x 最小值
    # print(min_x)
    return min_f,min_x

    # 画图
    '''x_label = np.arange(0, generation + 1, 1)
    plt.plot(x_label, min_f, color='blue')
    plt.xlabel('iteration')
    plt.ylabel('fx')
    plt.savefig('./iteration-f.png')
    plt.show()'''

if __name__=="__main__":
    T = 1
    t = np.zeros(T)
    value = np.zeros(T)
    for i in range(T):
        timestart = time.time()


        NP = 50  # 种群数量
        F = 0.6  # 缩放因子
        CR = 0.7  # 交叉概率
        generation = 200  # 遗传代数
        len_x = 5  # 维度
        value_up_range = 1
        value_down_range = 0

        value[i] ,position= DE(NP, F, CR, generation, len_x, value_up_range, value_down_range)
        print(i, "i", "  ", value[i], )
        timeend = time.time()
        print(timeend-timestart,"时间")



    print("平均值：", np.average(value))
    print("最优值：", np.min(value))
    print("最差值：", np.max(value))