# coding: utf-8
import sys, os
import math
path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(path, os.pardir))
import numpy as np
from lib.mnist import load_mnist
from lib.layers import Dense, Relu, SoftmaxWithLoss
from matplotlib.pylab import plt

class Network1:
    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def loss(self, t):
        return self.layers[-1].loss(t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis = 1)
        if t.ndim != 1 : t = np.argmax(t, axis = 1)
        return np.sum(y == t) / float(x.shape[0])

    def forward_pass(self, x):
        self.predict(x)

    def backward_pass(self, learning_rate):
        d_out = 1
        for layer in reversed(self.layers):
            d_out = layer.backward(d_out, learning_rate) # backward pass

    def evaluate(self, x_test, t_test):
        test_acc = self.accuracy(x_test, t_test)
        print("Test accuracy={0}".format(test_acc))

    def train(self, x_train, t_train):
        batch_size = 128
        epoches = 20
        train_size = x_train.shape[0]
        learning_rate = 0.1
        train_errors = []
        train_acc_list = []
        iter_per_epoch = int(math.ceil(train_size / batch_size))
        for epoch in range(1, epoches + 1):
            print("Epoch {0}/{1}".format(epoch, epoches))
            for _ in range(iter_per_epoch):
                batch_mask = np.random.choice(train_size, batch_size)
                x_batch = x_train[batch_mask]
                t_batch = t_train[batch_mask]
                
                self.forward_pass(x_batch)
                loss = self.loss(t_batch)
                train_errors.append(loss)
                self.backward_pass(learning_rate)
            train_acc = self.accuracy(x_train, t_train)
            train_acc_list.append(train_acc)
            print("Train accuracy={0}".format(train_acc))
        return train_errors
        
    def plot_error(self, train_errors):
        n = len(train_errors)
        training, = plt.plot(range(n), train_errors, label="Training Error")
        plt.legend(handles=[training])
        plt.title("Error Plot")
        plt.ylabel('Error')
        plt.xlabel('Iterations')
        plt.show()

    def show_failures(self, x_test, t_test):
        y = self.predict(x_test)
        y = np.argmax(y, axis = 1)
        if t_test.ndim != 1 : t_test = np.argmax(t_test, axis = 1)
        failures = []
        for idx in range(x_test.shape[0]):
            if y[idx] != t_test[idx]:
                failures.append((x_test[idx], y[idx], t_test[idx]))
        for i in range(min(len(failures), 60)):
            img, y, _ = failures[i]
            if (i % 10 == 0) : print()
            print(y, end=", ")
            img = img.reshape(28, 28)
            plt.subplot(6, 10, i + 1)
            plt.imshow(img, cmap='gray')
        print()
        plt.show()


(x_train, t_train), (x_test, t_test) = load_mnist()
network = Network1()
network.add(Dense(784, 50))
network.add(Relu())
network.add(Dense(50, 10))
network.add(SoftmaxWithLoss())

errors = network.train(x_train, t_train)
network.plot_error(errors)
network.evaluate(x_test, t_test)
network.show_failures(x_test, t_test)


