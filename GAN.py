# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
import os
import numpy as np
import random
import tensorflow as tf
import pickle
import time
# Graph
import matplotlib.pyplot as plot

class DataDistribution(object):
    def __init__(self):
        self.mu = 4
        self.sigma = 0.5

    def sample(self, N):
        samples = np.random.normal(self.mu, self.sigma, N)
        #samples.sort()
        return samples

class GeneratorDistribution(object):
    def __init__(self, range):
        self.range = range

    def sample(self, N):
        return np.linspace(-self.range, self.range, N) + \
            np.random.random(N) * 0.01

TOY_DATA = DataDistribution()
GEN_DATA = GeneratorDistribution(8)
x = [i/100. for i in range(0,1600)]
y = TOY_DATA.sample(1600)
g = GEN_DATA.sample(1600)

fig = plot.figure()
ax1 = fig.add_subplot(111)
ax1.scatter(x=x, y=y, c='blue',marker='o',s=10,label="Sample Toy Data")
ax1.scatter(x=x, y=g, c='red',marker='o',s=10,label="Sample Generator Data")
plot.show()
