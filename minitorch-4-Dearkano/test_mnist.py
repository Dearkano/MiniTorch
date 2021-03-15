from mnist import MNIST
import numpy as np
import matplotlib.pyplot as plt
mndata = MNIST("data/")
images, labels = mndata.load_training()
im = np.array(images[0])
im = im.reshape(28, 28)
print(im)
plt.imshow(im)
print('test')
