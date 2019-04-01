import numpy as np

a = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
        [10,11,12]
    ])
print(a)
b = a.copy()    # Deep copy

print("Integer array indexing")
print(a[[0,1,2,3], [2, 1, 1, 0]])
print(a[range(4), [0, 2, 0, 1]])
a[range(4), [0,1,1,2]] *= 2
print(a)

print("Boolean array indexing")
print(b)
bool_idx = (b % 2 == 0)
print(bool_idx)
print(b[bool_idx])
print(b[b  > 6])

# Data types
x = np.array([1, 2])
print(x.dtype)

x = np.array([0.5, 0.3])
print(x.dtype)

x = np.array([1, 2], dtype=np.float32)
print(x.dtype)
