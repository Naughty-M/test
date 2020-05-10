import copy
import math

import numpy as np


def sigmoid(x, b, k):
    try:
        result = 1 / (1 + math.e ** (-b * (x - k)))
        return result
    except OverflowError:
        if(x>k):
            return 1.0
        else:
            return 0.0


def MSE(x1, x2):
    # x_1 = np.array(x1)
    # x_2 = np.array(x2)
    len_len = len(x1)
    result = 0.
    for i in range(len_len):
        result += (x1[i] - x2[i]) ** 2

    return result / len_len


def Funvtion(position):
    """

    :param x_: position
    :param list: original input
    :return: fitness
    """
    # ww = [[0.2, 0.5, 0.3, 0.4, 0.6], [0.2, 0.5, 0.3, 0.4, 0.6], [0.2, 0.5, 0.3, 0.4, 0.6], [0.2, 0.5, 0.3, 0.4, 0.6],
    #       [0.2, 0.5, 0.3, 0.4, 0.6]]
    # uu = [[0.7, 0.60132241, 0.6, 0.8, 0.7, ], [0.7, 0.60132241, 0.6, 0.8, 0.7, ], [0.7, 0.60132241, 0.6, 0.8, 0.7, ],
    #       [0.7, 0.60132241, 0.6, 0.8, 0.7], [0.7, 0.60132241, 0.3885446, 0.8, 0.7]]
    # tt = [[0.3, 0.4, 0.2, 0.5, 0.4, ], [0.29999992, 0.39999976, 0.1999689, 0.5, 0.4, ],
    #       [0.30000007, 0.3999999, 0.20003368, 0.5, 0.4], [0.30000006, 0.40000026, 0.19995636, 0.5, 0.4],

    #       [0.18344421, 0.39814048, 0.19650983, 0.5000108, 0.400008]]
    # w = [0] + ww[0]  # [0., 0.2, 0.5, 0.3, 0.4, 0.6]
    # u = [0] + uu[0]  # [0., 0.7, 0.60132241, 0.6 , 0.8, 0.7]
    # t = [0] + tt[0]  # [0., 0.3, 0.4, 0.2, 0.5, 0.4]

    # w = [0] + [0.2223, 0.5062, 0.2731, 0.3798, 0.5947, 0.33912]
    # u = [0] + [0.6860, 0.8136, 0.6457, 0.8440, 0.7105, 2.36121]
    # t = [0] + [0.2796, 0.4419, 0.3009, 0.4814, 0.4240, 2.65551]
    D = len(position)
    # input_list = [[0.9,0.9,0.9,0.9],[0.8,0.8,0.8,0.8],[0.7,0.7,0.7,0.7],
    #               [0.8,0.9,0.9,0.9],[0.9,0.8,0.9,0.9],[0.7,0.7,0.9,0.9],
    #               [0.9,0.9,0.8,0.8],[0.9,0.8,0.6,0.9],[0.7,0.6,0.8,0.8],
    #               [0.9,0.8,0.7,0.6],[0.5,0.3,0.4,0.8],[0.2,0.8,0.7,0.3],
    #               [0.5,0.4,0.6,0.8],[0.2,0.6,0.7,0.8],[0.5,0.4,0.9,0.8]]
    input_list = [[0.9,0.9,0.9,0.9],[0.2,0.8,0.5,0.8],[0.4,0.7,0.6,0.7],
                  [0.8,0.4,0.2,0.9],[0.9,0.5,0.4,0.9],[0.3,0.7,0.5,0.9]]
    b = 10
    result = 0
    # w= [0]+[0.02253573 ,0.35688365, 0.12253573, 0.3366199,  0.54321143]
    # for  x in range(0,len(input_list)):
    for x in range(0,5):
        p1 = input_list[x][0]
        p4 = input_list[x][1]
        p5 = input_list[x][2]
        p7 = input_list[x][3]

        getthep8 = return_p8(input_list[x])  #p8 of the input sample

        for i in range(D):
            w = [0., 0.2, 0.5, 0.3, 0.4, 0.6]
            u = [0., 0.7, 0.9, 0.6, 0.8, 0.7]
            t = [0., 0.3, 0.4, 0.2, 0.5, 0.4]
            if (i < 5):
                w[i + 1] = position[i] + 0.0
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
                result += (p8 - getthep8) ** 2

            elif i < 10:
                u[i + 1 - 5] = position[i] + 0.0
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
                result += (p8 - getthep8) ** 2
            else:
                t[i + 1 - 10] = position[i] + 0.0
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
                result += (p8 - getthep8) ** 2
        # # print(p8-g)
    return  result

def return_p8(list_data):

    w = [0]+[0.2, 0.5, 0.3, 0.4, 0.6]
    u = [0]+[0.7, 0.9, 0.6, 0.8, 0.7]
    t = [0]+[0.3, 0.4, 0.2, 0.5, 0.4]

    # w= [0]+[0.02253573 ,0.35688365, 0.12253573, 0.3366199,  0.54321143]
    p1 = list_data[0]
    p4 = list_data[1]
    p5 = list_data[2]
    p7 = list_data[3]
    b = 10
    p9 = p1 * u[1] * sigmoid(p1, b, t[1])
    # print(p9,"P_9")
    p2 = p1 * u[2] * sigmoid(p1, b, t[2])
    x1 = p9
    x2 = p2 * u[3] * sigmoid(p2, b, t[3])
    # p3 = max(x1, x2)
    p3 = x1 * sigmoid(x1, b, x2) + x2 * sigmoid(x1, b, x2)
    x3 = p4 * w[1] + p3 * w[2] + p5 * w[3]
    p6 = x3 * u[4] * sigmoid(x3, b, t[4])
    x4 = p6 * w[4] + p7 * w[5]
    p8 = x4 * u[5] * sigmoid(x4, b, t[5])
    # print(p8,"--------return  sample p8")
    return  p8

if __name__ == "__main__":
    # print(Funvtion([]))

    # list_data = [0.9,0.9,0.9,0.9]

    set_list= [[0.9,0.9,0.9,0.9],[0.8,0.8,0.8,0.8],[0.7,0.7,0.7,0.7],
               [0.8,0.9,0.9,0.9],[0.9,0.8,0.9,0.9],[0.7,0.7,0.9,0.9],
               [0.9,0.9,0.8,0.8],[0.9,0.8,0.6,0.9],[0.7,0.6,0.8,0.8],
               [0.9,0.8,0.7,0.6],[0.5,0.3,0.4,0.8],[0.2,0.8,0.7,0.3],
               [0.5,0.4,0.6,0.8],[0.2,0.6,0.7,0.8],[0.5,0.4,0.9,0.8]] #样本

    x = [0.2,0.4,0.6,0.8,0.5]
    print(Funvtion(x))
    # x1= [0.5680152033178332]
    # x2= [0.568015203115994]
    # print(MSE(x1,x2))
