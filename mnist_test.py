import numpy as np
import tflearn.datasets.mnist as mnist
from apf import LinearEncoder

# Download mnist dataset, if not already owned, and extract it
train_x, train_y, test_x, test_y = mnist.load_data(one_hot=True)

print(np.shape(train_x))

# autoencoder = LinearEncoder()