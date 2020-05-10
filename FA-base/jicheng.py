import math
import time

import numpy as np
from FA import FA
import matplotlib.pyplot as plt
from DE import DE
from  PSO import function_PSO
from FA_oragin import FA_orgain

def shw_chart(teration,*para):
    b = [0] * teration
    for i in range(teration):
        b[i] = i
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_ylim([0, 10**(0)])
    ax.plot(b,para[0],label="CBOFA",linestyle="-."  )
    ax.plot(b,para[1],label="DE",   linestyle=":"   )
    ax.plot(b,para[2],label="PSO",  linestyle="--"  )
    # ax.plot(b,para[3],label="FA")
    plt.ylabel("MSE", fontsize=14)
    plt.xlabel("Iteration", fontsize=14)
    plt.legend()
    plt.show()

def MSN(list_1,list_original,the_time,dim):
    return_list=[]
    for t in range(the_time):
        len_len = dim
        result = 0.
        for i in range(len_len):
            result += (list_1[t][i] - list_original[i]) ** 2
        return_list.append(result / len_len)

    return return_list

if __name__=="__main__":

    dim = 15
    Teration = 200
    time_start = time.time()
    fa = FA(dim, 50, 1, 1.5, 0.99, Teration, [0.1, 1], 5)
    FA_y ,FA_position = fa.copy_iterate()
    time_end = time.time()
    FA_time = time_end-time_start
    print("FA --- OK ")

    #  DE()
    time_start = time.time()
    NP = 50  # 种群数量
    F = 0.7 # 缩放因子
    CR = 0.6  # 交叉概率
    generation = Teration  # 遗传代数
    len_x = dim  # 维度
    value_up_range = 1
    value_down_range = 0
    DE_y ,DE_position= DE(NP, F, CR, generation, len_x, value_up_range, value_down_range)
    print("DE-------OK ")
    time_end = time.time()
    DE_time = time_end - time_start

    #PSO
    #dim, size, iter_num, x_max, max_vel
    time_start = time.time()
    PSO_y,PSO_position = function_PSO(dim,50,Teration,1,0.9)
    time_end = time.time()
    PSO_time = time_end - time_start
    print("PSO ----OK")

    # fa_orgain = FA_orgain(30, 30, 1, 1.0, 0.7, 500, [-100, 100])
    # FA_oragin_y = fa_orgain.iterate()
    # print("SFA",FA_y)
    # print("DE_y",DE_y)
    # print("PSO_y",PSO_y)

    # print("FA_position",FA_position)
    # print("DE_position",DE_position)
    # print("PSO_posion",PSO_position)
    # # print(FA_oragin_y,"FA_oragin_y")
    # # print(FA_y[:200],"\n",DE_y[:200],"\n",PSO_y[:200])
    print("postion[-1]")
    print(FA_position [-1])
    print(DE_position [-1])
    print(PSO_position[-1])
    print("fitness")
    print(FA_y[-1])
    print(DE_y[-1])
    print(PSO_y[-1])
    print("time")
    print(FA_time)
    print(DE_time)
    print(PSO_time)

    w = [0.2, 0.5, 0.3, 0.4, 0.6]
    u = [0.7, 0.9, 0.6, 0.8, 0.7]
    t = [0.3, 0.4, 0.2, 0.5, 0.4]
    # #
    # FA_MSN = MSN(FA_position,  t, Teration, dim)
    # DE_MSN = MSN(DE_position,  t, Teration, dim)
    # PSO_MSN = MSN(PSO_position,t, Teration, dim)


    FA_MSN = MSN(FA_position,w+u+t,Teration,dim)
    DE_MSN = MSN(DE_position, w+u+t, Teration, dim)
    PSO_MSN = MSN(PSO_position, w+u+t, Teration, dim)
    print("MSE")
    print(FA_MSN [-1])
    print(DE_MSN [-1])
    print(PSO_MSN[-1])

    print(FA_MSN)
    print(DE_MSN)
    print(PSO_MSN)

    # shw_chart(200,FA_y[:200],DE_y[:200],PSO_y[:200])
    shw_chart(Teration,FA_MSN[:Teration],DE_MSN[:Teration],PSO_MSN[:Teration])
