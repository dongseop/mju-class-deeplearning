import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / np.sum(e)

def mean_squared_error(y, t):
    return np.mean((y - t) ** 2)

def cross_entropy_error(y, t):
    return -np.sum(t * np.log(y + 1e-9)) / y.shape[0]

# t = np.array([1, 0, 0, 0, 0])
# y1 = np.array([0.4, 0.3, 0.1, 0.1, 0.1])
# y2 = np.array([0.35, 0.01, 0.01, 0.01, 0.01])

# print(mean_squared_error(y1, t), mean_squared_error(y2, t))
# print(cross_entropy_error(y1, t), cross_entropy_error(y2, t))


