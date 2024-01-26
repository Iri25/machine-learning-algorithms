import matplotlib.pyplot as plt
import numpy as np
import sklearn
from mpl_toolkits.mplot3d import Axes3D


def plotDataHistogram(x, variableName):
    n, bins, patches = plt.hist(x, 10)
    plt.title('Histogram of ' + variableName)
    plt.show()


def plotData(x1, y1, x2=None, y2=None, x3=None, y3=None, title=None):
    plt.plot(x1, y1, 'ro', label='train data')
    if x2:
        plt.plot(x2, y2, 'b-', label='learnt model')
    if x3:
        plt.plot(x3, y3, 'g^', label='test data')
    plt.title(title)
    plt.legend()
    plt.show()
