import sys, os
from lib.functions import sigmoid, softmax, cross_entropy_error, mean_squared_error
import numpy as np 

class Dense:
    def __init__(self, input_size, output_size, initializer='random'):
        self.W = 0.1 * np.random.randn(input_size, output_size)
        self.b = 0.1 * np.zeros(output_size)
        self.x = None
        self.y = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        self.y = np.dot(self.x, self.W) + self.b
        return self.y

    def backward(self, d_out, learning_rate):
        self.dW = np.dot(self.x.T, d_out)
        self.db = np.sum(d_out, axis = 0)
        d_x = np.dot(d_out, self.W.T)
        self.W -= learning_rate * self.dW
        self.b -= learning_rate * self.db
        return d_x


class SoftmaxWithLoss:
    def __init__(self):
        self.error = None
        self.y = None
        self.t = None

    def forward(self, x):
        self.y = softmax(x)
        return self.y

    def loss(self, t):
        self.t = t
        self.error = cross_entropy_error(self.y, self.t)
        return self.error

    def backward(self, d_out = 1, learning_rate = None):
        batch_size = self.t.shape[0]
        d_x = (self.y - self.t) / batch_size
        return d_x


class Sigmoid:
    def __init__(self):
        self.y = None

    def forward(self, x):
        self.y = sigmoid(x)
        return self.y

    def backward(self, d_out, learning_rate = None):
        return d_out * (1.0 - self.y) * self.y


class Relu:
    def __init__(self):
        self.mask = None
    
    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, d_out, learning_rate = None):
        d_out[self.mask] = 0
        d_x = d_out
        return d_x