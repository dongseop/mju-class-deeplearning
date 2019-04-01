import numpy as np

x = np.array([[1,2],[3,4]], dtype=np.float64)
y = np.array([[5,6],[7,8]], dtype=np.float64)

print(x + y)
print(np.add(x, y))

print(x - y)
print(np.subtract(x, y))

print(x * y)
print(np.multiply(x, y))

print(x / y)
print(np.divide(x, y))

print(np.sqrt(x))

print(y**2)


print(x @ y)
print(x.dot(y))
print(np.dot(x, y))

print(x.transpose())
print(x.T)

print(np.linalg.inv(y))
z = np.eye(3)
print(z)
print(np.linalg.inv(z))

print()

v = np.array([1,2,3]) 
w = np.array([4,5])   
print(np.reshape(v, (3, 1)) * w)
print(np.outer(v, w))

x = np.array([[1,2,3], [4,5,6]])
print(x + v)

print((x.T + w))
print(x + np.reshape(w, (2, 1)))

print(x * 2)