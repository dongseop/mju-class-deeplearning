import numpy as np

a = np.array([1, 2, 3])
print(a)
print(type(a))
print(a.shape)
print(a[0], a[1], a[2])
a[0] = 5
print(a)
print(a.ndim)
b = np.array([[1,2,3],[4,5,6]])    
print(b.shape)                     
print(b) 
print(b.ndim)

# reshape
a = np.arange(1, 10)
print(a)
a.shape = 3, 3
print(a)
print(np.arange(15).reshape(3, 5))
