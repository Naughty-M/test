import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':




    x = np.random.random([1,2])*100
    print(x)

    for i in range(100):
        # print(np.random.random([1,2])*100)
        x = np.append(x,np.random.random([1,2])*100,axis=0)

    print(x)
    plt.axis([-100, 100, -100, 100])

    plt.scatter(x[:,0],x[:,1],c='r')
    plt.show()
    # plt.figure()

