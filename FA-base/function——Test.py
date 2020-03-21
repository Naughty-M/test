
import math


def fan(T,t):
    resule = 1/(1+math.exp((math.log(3)+math.log(99))*t/T-math.log(99)))
    return resule

if __name__ == '__main__':
    T = 10

    for t in range(T):
        print(T,t)
        print(fan(T,t+1))