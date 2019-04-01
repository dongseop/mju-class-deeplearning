import numpy as np
from mnist_data import load_mnist
from functions import sigmoid, softmax

layers = [784, 20, 10, 10]

def init_network():
    network = {}
    network['W1'] = 0.01 * np.random.randn(layers[0], layers[1])
    network['W2'] = 0.01 * np.random.randn(layers[1], layers[2])
    network['W3'] = 0.01 * np.random.randn(layers[2], layers[3])
    network['b1'] = np.zeros(layers[1])
    network['b2'] = np.zeros(layers[2])
    network['b3'] = np.zeros(layers[3])
    return network

def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    x1 = sigmoid(np.dot(x, W1) + b1)
    x2 = sigmoid(np.dot(x1, W2) + b2)
    x3 = np.dot(x2, W3) + b3
    y = softmax(x3)
    return y

def accuracy(network, x, t):
    y = predict(network, x)
    print(y, t)
    y = np.argmax(y, axis=1)
    t = np.argmax(t, axis=1)
    accuracy = np.sum(y == t) / float(x.shape[0])
    return accuracy

(x_train, y_train), (x_test, y_test) = load_mnist()
network = init_network()
print(accuracy(network, x_train, y_train))
    