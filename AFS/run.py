import numpy as np
from AFS import *

if __name__ == "__main__":
    t = np.zeros(10)
    value = np.zeros(10)
    for i in range(30):

        bound = np.tile([[-100], [100]], 30)
        # (self, sizepop, vardim, bound, MAXGEN, params):
        afs = ArtificialFishSwarm(30, 30, bound, 200, [0.001, 0.0001, 0.618, 40])
        value[i] = afs.solve()

    print("平均值：", np.average(value))
    print("最优值：", np.min(value))
    print("最差值：", np.max(value))