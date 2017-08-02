import numpy as np
import tflearn
import tflearn.datasets.mnist as mnist
from apf import LinearEncoder

# Download mnist dataset, if not already owned, and extract it
train_x, train_y, test_x, test_y = mnist.load_data(one_hot=True)

# print(np.shape(train_x)) == (55000, 784)
SQUARE_ROOT_OF_INPUT_SIZE = 28
autoencoder = LinearEncoder(input_size=784,
                            output_size=SQUARE_ROOT_OF_INPUT_SIZE,
                            model_id="mnist_linear_encoder_test")

# Test successful training
autoencoder.train(train_x,
                  batch_size=100,
                  epochs=5)

# Test encoding abilities
encoder_model = tflearn.DNN(autoencoder.encoder,
                            session=autoencoder.session)