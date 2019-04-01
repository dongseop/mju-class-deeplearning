import numpy as np 

def init_network():
    network = {}
    network['W'] = np.array([
        [0.2, 0.5, 0.3],
        [0.8, 0.6, 0.4]
    ])
    return network

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def forward(network, x):
    y = sigmoid(np.dot(x, network['W']))
    return y

def softmax(x):
    e = np.exp(x - np.max(x))
    s = np.sum(e)
    return e / s

# network = init_network()
# y = forward(network, np.array([1.0, 2.0]))
# print(y)


print(softmax(np.array([4, 0, 2, 9])))
print(softmax(np.array([10, 10, 10, 10])))
print(softmax(np.array([0, 0, 0, 10])))
print(softmax(np.array([200, 1000, 100, 200])))