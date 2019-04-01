import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / np.sum(e, axis=-1, keepdims=True)

def mean_squared_error(y, t):
    return np.mean((y - t) ** 2)

def cross_entropy_error(y, t):
    return -np.sum(t * np.log(y + 1e-9)) / y.shape[0]
    