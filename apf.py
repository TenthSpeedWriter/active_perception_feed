from uuid import uuid4
import tflearn

class InputStreamEncoder:
    """A base class for any model which
        1: Trains as an autoencoder, and
        2: Converts a given input format into a 1-dimensional output array
    """
    # Mostly writing this in case I want to make it an abstract

class LinearEncoder:
    def __init__(self, input_size, output_size,
                 model_id=None,
                 decoder_activation="sigmoid",
                 optimizer="adam",
                 learn_rate=0.001,
                 loss_func="mean_squared"):
        """

        :param input_size: Scalar integer value - size of the input array
        :param output_size: Scalar integer value - size of the output array
        :param model_id: TF model name for this instance. Defaults to a random UUID.
        :param decoder_activation: Name of the decoder activation function.
        :param optimizer: Defaults to the flexible Adam optimizer
        :param learn_rate: Learning rate for this model
        :param loss_func: Defaults to mean squared; use cross-entropy for probability inputs.
        """

        # Generate a random UUID to use as the model ID if one is not provided.
        self.model_id = model_id if model_id is not None else uuid4()

        # Establish input parameters: [Any batch size, specific record size]
        self.encoder = tflearn.input_data(shape=[None, input_size])

        # Generate encoder layer
        self.encoder = tflearn.fully_connected(self.encoder,
                                               output_size)

        # Construct decoder by adding an output-shaped fully connected layer
        # and, atop that, a sigmoid-wrapped input-shaped layer.
        self.decoder = tflearn.fully_connected(self.encoder,
                                               input_size,
                                               activation=decoder_activation)

        self.net = tflearn.regression(self.decoder,
                                      optimizer=optimizer,
                                      learning_rate=learn_rate,
                                      loss=loss_func)

    def train(self, train_data,
              batch_size=10,
              epochs=3,
              tensorboard_verbose=0):
        """
        Train, from the decoder, the network's ability to process; understand;
        and recreate date fed to the encoder.

        :param train_data: A set from which to draw training data
        :param batch_size: Size of each individual training batch
        :param epochs: Number of epochs - times each training row is used
        :param tensorboard_verbose: Verbosity level (0 through 3)
        """
        run_id = "Input autoencoder training: " + str(self.model_id)

        self.model = tflearn.DNN(self.net,
                                 tensorboard_verbose=tensorboard_verbose)
        self.model.fit(train_data, train_data,
                       batch_size=batch_size,
                       n_epoch=epochs,
                       run_id=run_id)

# I'd like to make a conv2d option available, too!