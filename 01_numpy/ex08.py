import numpy as np
from scipy.spatial.distance import pdist, squareform

x = np.array([[0, 1], [1, 0], [2, 0]])
print(x)

d1 = pdist(x, 'euclidean')
print(d1)
print(squareform(d1))

d2 = pdist(x, 'cityblock')
print(squareform(d2))

d2 = pdist(x, 'cosine')
print(squareform(d2))

