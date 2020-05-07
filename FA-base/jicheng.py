import math

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
    ax.set_ylim([0, 10**(-2)])
    ax.plot(b,para[0],label="CBOFA",linestyle="-."  )
    ax.plot(b,para[1],label="DE",   linestyle=":"   )
    ax.plot(b,para[2],label="PSO",  linestyle="--"  )
    # ax.plot(b,para[3],label="FA")
    plt.ylabel("Fitness", fontsize=14)
    plt.xlabel("Iteration", fontsize=14)
    plt.legend()
    plt.show()

if __name__=="__main__":

    fa = FA(5, 50, 1, 1.1, 0.97, 200, [0, 1], 5)
    FA_y ,FA_position = fa.copy_iterate()
    print("FA --- OK ")

    #  DE()
    NP = 50  # 种群数量
    F = 0.6  # 缩放因子
    CR = 0.7  # 交叉概率
    generation = 200  # 遗传代数
    len_x = 5  # 维度
    value_up_range = 1
    value_down_range = 0
    DE_y ,DE_position= DE(NP, F, CR, generation, len_x, value_up_range, value_down_range)
    print("DE-------OK ")

    #PSO
    #dim, size, iter_num, x_max, max_vel
    PSO_y,PSO_position = function_PSO(5,50,200,1,0.5)

    print("PSO ----OK")

    # fa_orgain = FA_orgain(30, 30, 1, 1.0, 0.7, 500, [-100, 100])
    # FA_oragin_y = fa_orgain.iterate()
    print("SFA",FA_y)
    print("DE_y",DE_y)
    print("PSO_y",PSO_y)

    print("FA_position",FA_position)
    print("DE_position",DE_position)
    print("PSO_posion",PSO_position)
    # print(FA_oragin_y,"FA_oragin_y")
    # print(FA_y[:200],"\n",DE_y[:200],"\n",PSO_y[:200])

    shw_chart(200,FA_y[:200],DE_y[:200],PSO_y[:200])

