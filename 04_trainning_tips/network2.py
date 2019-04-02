# coding: utf-8
import sys, os
import math
path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(path, os.pardir))
import numpy as np
from lib.mnist import load_mnist
from lib.layers2 import Dense, Relu, SoftmaxWithLoss
from matplotlib.pylab import plt

class Network2:
    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def predict(self, x, bTrain=False):
        for layer in self.layers:
            x = layer.forward(x, bTrain)
        return x

    def loss(self, t):
        weight_decay = 0
        for layer in self.layers:
            if type(layer) is Dense:
                weight_decay += layer.weight_decay()
        return self.layers[-1].loss(t) + weight_decay

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis = 1)
        if t.ndim != 1 : t = np.argmax(t, axis = 1)
        return np.sum(y == t) / float(x.shape[0])

    def forward_pass(self, x, bTrain=False):
        self.predict(x, bTrain)

    def backward_pass(self, learning_rate):
        d_out = 1
        for layer in reversed(self.layers):
            d_out = layer.backward(d_out, learning_rate) # backward pass

    
    def evaluate(self, x_test, t_test):
        test_acc = self.accuracy(x_test, t_test)
        
        return test_acc

    def train(self, x_train, t_train, x_test, t_test):
        batch_size = 128
        epoches = 5
        train_size = x_train.shape[0]
        learning_rate = 0.1
        train_errors = []
        train_acc_list = []
        test_acc_list = []
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
                test_acc = self.evaluate(x_test, t_test)
                print("Train accuracy={0}".format(train_acc))
                print("Test accuracy={0}".format(test_acc))
                train_acc_list.append(train_acc)
                test_acc_list.append(test_acc)
        return train_errors, train_acc_list, test_acc_list
        
    def plot_accuracy(self, train_acc, test_acc):
        n = len(train_acc)
        plt.plot(range(n), train_acc, label="train")
        plt.plot(range(n), test_acc, label="test")
        plt.legend(loc='lower right')
        plt.title("Overfit")
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
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
options = {
    'weight_decay_lambda': 0,
    'learning_rate': 0.01, 
    'beta1': 0.9,
    'beta2': 0.999
}
overfitmask = np.random.choice(x_train.shape[0], 300)
x_train = x_train[overfitmask]
t_train = t_train[overfitmask]
network = Network2()
network.add(Dense(784, 50, options, 'sgd', 'he'))
network.add(Relu())
network.add(Dense(50, 10, options, 'sgd', 'xavier'))
network.add(SoftmaxWithLoss())

errors, train_acc, test_acc = network.train(x_train, t_train, x_test, t_test)
network.plot_accuracy(train_acc, test_acc)
# network.evaluate(x_test, t_test)
network.show_failures(x_test, t_test)


