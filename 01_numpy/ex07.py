import numpy as np
import matplotlib.pyplot as plt 
from scipy.misc import imread, imsave, imresize
img = imread('./img.jpg')

print(img)
print(img.dtype, img.shape)
img_tinted = img * [0.5, 1.0, 0.5]

plt.subplot(1, 2, 1)
plt.imshow(img)
plt.subplot(1, 2, 2)
plt.imshow(np.uint8(img_tinted))
plt.show()

img_resized = imresize(img, (64, 64))
imsave('./resized.jpg', img_resized)

