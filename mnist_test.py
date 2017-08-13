#import numpy as np
#import tflearn
import tflearn.datasets.mnist as mnist
from apf import LinearEncoder, Unicoder

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
                  epochs=1)

# Test encoding abilities
test_encoding = autoencoder.encode(train_x)

# Simulate multiple input streams
encoders = {
    "foo": autoencoder,
    "bar": autoencoder
}

# Use them as the basis for a unicoder
unicoder = Unicoder(encoders=encoders)

# Attempt unicoder training
unicoder.construct(sources={
    "foo": train_x,
    "bar": train_x
})
