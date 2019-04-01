import sys
import os
from lib.functions import sigmoid, softmax, cross_entropy_error, mean_squared_error
from lib.optimizers import SGD, Momentum, AdaGrad, RMSProp, Adam
import numpy as np


class Dense:
    def __init__(self, input_size, output_size, options = {}, optimizer='sgd', initializer='random'):
        if optimizer == 'sgd':
            self.optimizerW = SGD(options.get('learning_rate', 0.01))
            self.optimizerB = SGD(options.get('learning_rate', 0.01))
        elif optimizer == 'momentum':
            self.optimizerW = Momentum(
                options.get('learning_rate', 0.01), options.get('momentum', 0.9))
            self.optimizerB = Momentum(
                options.get('learning_rate', 0.01), options.get('momentum', 0.9))
        elif optimizer == 'adagrad':
            self.optimizerW = AdaGrad(options.get('learning_rate', 0.01))
            self.optimizerB = AdaGrad(options.get('learning_rate', 0.01))
        elif optimizer == 'rmsprop':
            self.optimizerW = RMSProp(options.get(
                'learning_rate', 0.01), options.get('decay_rate', 0.9))
            self.optimizerB = RMSProp(options.get(
                'learning_rate', 0.01), options.get('decay_rate', 0.9))
        elif optimizer == 'adam':
            self.optimizerW = Adam(
                options.get('learning_rate', 0.001), options.get('beta1', 0.9),
                options.get('beta2', 0.999))
            self.optimizerB = Adam(
                options.get('learning_rate', 0.001), options.get('beta1', 0.9),
                options.get('beta2', 0.999))

        if initializer == "random":
            self.W = 0.1 * np.random.randn(input_size, output_size)
        elif initializer == "xavier":
            """
            Glorot, Xavier, and Yoshua Bengio. 
            "Understanding the difficulty of training deep feedforward neural networks." 
            Proceedings of the thirteenth international conference on artificial intelligence and statistics. 2010.
            http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf?hc_location=ufi
            """
            self.W = np.random.randn(input_size, output_size) / np.sqrt(input_size)
        elif initializer == "he":
            """
            He, Kaiming, et al. 
            "Delving deep into rectifiers: Surpassing human-level performance on imagenet classification." 
            Proceedings of the IEEE international conference on computer vision. 2015.
            """
            self.W = np.random.randn(input_size, output_size) / np.sqrt(input_size / 2)

        self.b = 0.1 * np.zeros(output_size)

        self.x = None
        self.y = None
        self.dW = None
        self.db = None
        self.weight_decay_lambda = options.get('weight_decay_lambda', 0)

    def forward(self, x, bTrain):
        self.x = x
        self.y = np.dot(self.x, self.W) + self.b
        return self.y

    def weight_decay(self):
        return 0.5 * self.weight_decay_lambda * np.sum(self.W**2)

    def backward(self, d_out, learning_rate):
        self.dW = np.dot(self.x.T, d_out) + self.weight_decay_lambda * self.W
        self.db = np.sum(d_out, axis=0)
        d_x = np.dot(d_out, self.W.T)
        self.optimizerW.update(self.W, self.dW)
        self.optimizerB.update(self.b, self.db)
        return d_x


class SoftmaxWithLoss:
    def __init__(self):
        self.error = None
        self.y = None
        self.t = None

    def forward(self, x, bTrain):
        self.y = softmax(x)
        return self.y

    def loss(self, t):
        self.t = t
        self.error = cross_entropy_error(self.y, self.t)
        return self.error

    def backward(self, d_out=1, learning_rate=None):
        batch_size = self.t.shape[0]
        d_x = (self.y - self.t) / batch_size
        return d_x


class Sigmoid:
    def __init__(self):
        self.y = None

    def forward(self, x, bTrain):
        self.y = sigmoid(x)
        return self.y

    def backward(self, d_out, learning_rate=None):
        return d_out * (1.0 - self.y) * self.y


class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x, bTrain):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, d_out, learning_rate=None):
        d_out[self.mask] = 0
        d_x = d_out
        return d_x


