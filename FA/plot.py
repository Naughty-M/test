import matplotlib.pyplot as plt

def plot(X_origin, X):
    fig_origin = plt.figure(0)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.scatter(X_origin[:, 0], X_origin[:, 1], c='r')
    plt.scatter(X[:, 0], X[:, 1], c='g')
    plt.pause(0.1)
    plt.clf()

def plot(X_origin):
    fig_origin = plt.figure(0)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.scatter(X_origin[:, 0], X_origin[:, 1], c='r')
    # plt.scatter(X[:, 0], X[:, 1], c='g')
    plt.pause(0.1)
    plt.clf()