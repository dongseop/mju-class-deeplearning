import numpy as np 
from mnist_data import load_mnist
from matplotlib.pylab import plt

(x_train, y_train), (x_test, y_test) = load_mnist()

for i in range(10):
    img = x_train[i]
    label = np.argmax(y_train[i])
    print(label, end=", ")
    img = img.reshape(28, 28)
    plt.subplot(1, 10, i + 1)
    plt.imshow(img)

print()
plt.show()

