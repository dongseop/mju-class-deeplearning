import numpy as np

class Optimizer:
    def __init__(self, learning_rate=0.01):
        self.lr = learning_rate

    def update(self, weight, grad):
        weight

""" Stochastic gradient descent """
class SGD(Optimizer):
    def __init__(self, learning_rate=0.01):
        super().__init__(learning_rate)

    def update(self, weight, grad):
        weight -= self.lr * grad

""" Momentum
Rumelhart, David E.; Hinton, Geoffrey E.; Williams, Ronald J. (8 October 1986). 
"Learning representations by back-propagating errors". 
Nature. 323 (6088): 533–536. 
"""
class Momentum(Optimizer):
    def __init__(self, learning_rate=0.01, momentum=0.9):
        super().__init__(learning_rate)
        self.momentum = momentum
        self.v = None
    
    def update(self, weight, grad):
        if self.v is None:
            self.v = np.zeros_like(weight)
        
        self.v = self.momentum * self.v - self.lr * grad
        weight += self.v


""" AdaGrad (adaptive gradient algorithm) 
Duchi, John; Hazan, Elad; Singer, Yoram (2011). 
"Adaptive subgradient methods for online learning and stochastic optimization". 
JMLR. 12: 2121–2159.
"""
class AdaGrad(Optimizer):
    def __init__(self, learning_rate=0.01):
        super().__init__(learning_rate)
        self.cache = None

    def update(self, weight, grad):
        if self.cache is None:
            self.cache = np.zeros_like(weight)
        self.cache += grad ** 2
        weight -= self.lr * grad / (np.sqrt(self.cache) + 1e-7)


""" RMSProp (Root Mean Square Propagation)
Tieleman, Tijmen and Hinton, Geoffrey (2012). 
Lecture 6.5-rmsprop: Divide the gradient by a running average of its recent magnitude. 
COURSERA: Neural Networks for Machine Learning
"""
class RMSProp(Optimizer):
    def __init__(self, learning_rate=0.01, decay_rate=0.9):
        super().__init__(learning_rate)
        self.decay_rate = decay_rate
        self.cache = None

    def update(self, weight, grad):
        if self.cache is None:
            self.cache = np.zeros_like(weight)
        self.cache = self.decay_rate * self.cache + (1 - self.decay_rate) * (grad ** 2)
        weight -= self.lr * grad / (np.sqrt(self.cache) + 1e-7)


"""
Adam (Adaptive Moment Estimation)
Diederik, Kingma; Ba, Jimmy (2014). 
"Adam: A method for stochastic optimization". 
arXiv:1412.6980
https://arxiv.org/abs/1412.6980
"""
class Adam(Optimizer):
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None

    def update(self, weight, grad):
        if self.m is None:
            self.m = np.zeros_like(weight)
            self.v = np.zeros_like(weight)

        self.iter += 1
        lr_t = self.lr * np.sqrt(1.0 - self.beta2 ** self.iter) / (1.0 - self.beta1 ** self.iter)

        self.m += (1 - self.beta1) * (grad - self.m)
        self.v += (1 - self.beta2) * (grad ** 2 - self.v)
        weight -= lr_t * self.m / (np.sqrt(self.v) + 1e-7)
        
