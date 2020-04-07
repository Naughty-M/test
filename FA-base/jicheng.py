import math

import numpy as np
from FA import FA
import matplotlib.pyplot as plt
from DE import DE
from  PSO import function_PSO
from FA_oragin import FA_orgain

def shw_chart(*para):
    b = [0] * 500
    for i in range(500):
        b[i] = i
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_ylim([0, 5])
    ax.plot(b,para[0],label="CBOFA",linestyle="-."  )
    ax.plot(b,para[1],label="DE",   linestyle=":"   )
    ax.plot(b,para[2],label="PSO",  linestyle="--"  )
    # ax.plot(b,para[3],label="FA")
    plt.ylabel("Fitness", fontsize=14)
    plt.xlabel("Iteration", fontsize=14)
    plt.legend()
    plt.show()

if __name__=="__main__":
    fa = FA(2, 30, 1, 1.0, 0.97, 500, [-65, 65], 5)
    FA_y = fa.copy_iterate()
    DE_y = DE()[:500]
    PSO_y = function_PSO()[:500]
    # fa_orgain = FA_orgain(30, 30, 1, 1.0, 0.7, 500, [-100, 100])
    # FA_oragin_y = fa_orgain.iterate()
    print(DE_y,'DE_y')
    print("PSO_y",PSO_y,)
    # print(FA_oragin_y,"FA_oragin_y")
    shw_chart(FA_y,DE_y,PSO_y)

