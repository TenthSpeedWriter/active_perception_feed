from uuid import uuid4
import tensorflow as tf
import tflearn


class InputStreamEncoder:
    """A base class for any model which
        1: Trains as an autoencoder, and
        2: Converts a given input format into a 1-dimensional output array
    """



class LinearEncoder(InputStreamEncoder):
    """Autoencodes a (None, input_size) shaped tensor."""
    model = None
    session = None
    def __init__(self, input_size, output_size,
                 model_id=None,
                 encoder_activation="sigmoid",
                 decoder_activation="sigmoid",
                 optimizer="adam",
                 learn_rate=0.001,
                 loss_func="mean_square",
                 train_dropout=0.25):
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
        self.input_size = input_size
        self.output_size = output_size

        with tf.name_scope(self.model_id):
            # Establish input parameters: [Any batch size, specific record size]
            self.input = tflearn.input_data(shape=[None, input_size])

            # Generate encoder layer
            self.encoder = tflearn.fully_connected(self.input, output_size,
                                                   activation=decoder_activation)

            # Construct decoder by adding an output-shaped fully connected layer
            # and, atop that, a sigmoid-wrapped input-shaped layer.
            self.decoder = tflearn.fully_connected(self.encoder, input_size,
                                                   activation=decoder_activation)

            self.net = tflearn.regression(self.decoder,
                                          optimizer=optimizer,
                                          learning_rate=learn_rate,
                                          loss=loss_func)

    def encode(self, X):
        """
        Encode a given data batch X whose shape corresponds to that of the
        input placeholder.

        Credit to Discharged Spider on StackOverflow for the effective
        approach to encoding and decoding using TF Learn.

        :param X: A batch of input data
        :return: Y, a set of encoded data corresponding to X.
        """
        # If given only a single record, cast it to shape (1, input_size)
        input_data = X if len(X.shape) >= 1 else X.reshape(1, -1)
        with tf.name_scope(self.model_id):
            with self.session.as_default():
                tflearn.is_training(False, self.session)
                return self.encoder.eval(feed_dict={
                    self.input: input_data
                })

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

        with tf.name_scope(self.model_id):
            self.model = tflearn.DNN(self.net,
                                     tensorboard_verbose=tensorboard_verbose)
            # Add a shorter reference to self.model.session, especially
            # considering it will also be the default session for the class.
            self.session = self.model.session

            # And of course, train the model.
            self.model.fit(train_data, train_data,
                           batch_size=batch_size,
                           n_epoch=epochs,
                           run_id=run_id)

# class Conv2dEncoder(InputStreamEncoder):
#    """
#    Keep in mind for training, it's the kernel that needs to be able to
#    recreate its own input.
#    """


class Unicoder:
    """
    An autoencoder which accepts multiple InputStreamEncoders and trains on
    their collective output at matching time segments.

    In application, this can be used to unify the encodings of multiple
    InputStreamEncoders into a single Active Perception Feed. Our objective
    is to use this as a means of presenting scenarios-over-time to higher
    level networks such as LSTM networks and Q-nets.

    Not to be confused with something which converts things to UTF-8.
    """
    def __init__(self, encoders,
                 model_id=None,
                 output_size=None):
        self.encoders = encoders

        # Use a random UUID as the model ID if one is not specified
        self.model_id = model_id if model_id is not None else uuid4()

        # The encoded data will be a number of (None, n) arrays with varying n.
        # Therefore, the input shape will be (None, sum(n_0 ... n_i))
        self.input_shape = (None,
                            sum([e.output_size
                                 for e in self.encoders.iiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiikkkkoooooooooooooooooooitems()]))
        self.input = tflearn.input_data(shape=self.input_shape)

        # If no output size is specified, default to the rounded 3/4ths root
        # of the input size. There's nothing magical about that value; it's
        # just a convenient default for a value you should really specify.
        self.output_size = output_size if output_size is not None else int(self.input_shape[1]^0.75)

        self.encoder = tflearn.fully_connected(self.input, self.output_size,
                                               activation="softmax")


    # def train_encoders(self, sources):
    #     """
    #     :param sources: A dict containing .train() input for each encoder.
    #                     Must be likewise-named. encoder.train() should be
    #                     implemented as per LinearEncoder.
    #     """
    #     # For each source name and values dict pair, execute the .train method
    #     # of the corresponding encoder using any kwargs previously packed
    #     # in the same section of the sources dictionary. train_data must
    #     # be among these.
    #     [self.encoders[name].train(**values) for name, values in sources]

    def construct(self, sources):
        """
        Initializes and trains the unicoder itself. Should be called upon or
        immediately after instantiation.

        :param sources: A dict with k/v pairs of "encoder_name": training_data
        """
        # Match each encoder with its input
        encoders_and_sources = [(encoder, sources[name])
                                for name, encoder in self.encoders.items]

        # Use each to encode its respective training data
        input_encodings = [encoder.encode(source)
                           for encoder, source in encoders_and_sources]




        with tf.name_scope(self.model_id):
            pass
            #self.model =
            #self