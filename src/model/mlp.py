
import numpy as np

from sklearn.metrics import accuracy_score

# from util.activation_functions import Activation
from util.loss_functions import CrossEntropyError
from model.logistic_layer import LogisticLayer
from model.classifier import Classifier


class MultilayerPerceptron(Classifier):
    """
    A multilayer perceptron used for classification
    """

    def __init__(self, train, valid, test, layers=None, encoding_layer=None, input_weights=None,
                 output_task='classification', output_activation='softmax',
                 cost='crossentropy', learning_rate=0.01, epochs=50):

        """
        A digit-7 recognizer based on logistic regression algorithm

        Parameters
        ----------
        train : list
        valid : list
        test : list
        learning_rate : float
        epochs : positive int

        Attributes
        ----------
        training_set : list
        validation_set : list
        test_set : list
        learning_rate : float
        epochs : positive int
        performances: array of floats
        encoding_layer: hidden layer learned from autoencoder
        """

        self.learning_rate = learning_rate
        self.epochs = epochs
        self.output_task = output_task  # Either classification or regression
        self.output_activation = output_activation
        self.cost = cost
        # Should polish the loss_function a little bit more
        self.error = CrossEntropyError

        self.training_set = train
        self.validation_set = valid
        self.test_set = test

        # Record the performance of each epoch for later usages
        # e.g. plotting, reporting..
        self.performances = []

        self.layers = layers
        self.input_weights = input_weights

        # Build up the network from specific layers
        if layers is None:
            self.layers = []

            # First hidden layer
            number_of_1st_hidden_layer = 100

            if encoding_layer is not None:
                self.layers.append(LogisticLayer(train.input.shape[1],
                                                 encoding_layer.shape[1],
                                                 activation="sigmoid",
                                                 is_classifier_layer=False,
                                                 weights=encoding_layer,
                                                 fixed_weights=True))

                self.layers.append(LogisticLayer(encoding_layer.shape[1],
                                                 number_of_1st_hidden_layer, None,
                                                 activation="sigmoid",
                                                 is_classifier_layer=False))
            else:
                self.layers.append(LogisticLayer(train.input.shape[1],
                                                 number_of_1st_hidden_layer, None,
                                                 activation="sigmoid",
                                                 is_classifier_layer=False))

            # Output layer
            self.layers.append(LogisticLayer(number_of_1st_hidden_layer,
                                             10, None,
                                             activation="softmax",
                                             is_classifier_layer=True))

        else:
            self.layers = layers

        # add bias values ("1"s) at the beginning of all data sets
        self.training_set.input = np.insert(self.training_set.input, 0, 1,
                                            axis=1)
        self.validation_set.input = np.insert(self.validation_set.input, 0, 1,
                                              axis=1)
        self.test_set.input = np.insert(self.test_set.input, 0, 1, axis=1)

    def _get_layer(self, layer_index):
        return self.layers[layer_index]

    def _get_input_layer(self):
        return self._get_layer(0)

    def _get_output_layer(self):
        return self._get_layer(-1)

    def _feed_forward(self, inp):
        """
        Do feed forward through the layers of the network

        Parameters
        ----------
        inp : ndarray
            a numpy array containing the input of the layer
        """

        # Feed forward layer by layer
        # The output of previous layer is the input of the next layer
        last_layer_output = inp

        for layer in self.layers:
            last_layer_output = layer.forward(last_layer_output)
            # Do not forget to add bias for every layer
            last_layer_output = np.insert(last_layer_output, 0, 1, axis=0)

    def _compute_error(self, target):
        """
        Compute the total error of the network

        Returns
        -------
        ndarray :
            a numpy array (1,nOut) containing the output of the layer
        """

        # Get output layer
        output_layer = self._get_output_layer()

        # Calculate the deltas of the output layer
        output_layer.deltas = target - output_layer.outp

        # Calculate deltas (error terms) backward except the output layer
        for i in reversed(range(0, len(self.layers) - 1)):
            current_layer = self._get_layer(i)
            next_layer = self._get_layer(i+1)
            next_weights = np.delete(next_layer.weights, 0, axis=0)
            next_derivatives = next_layer.deltas

            current_layer.computeDerivative(next_derivatives, next_weights.T)

    def _update_weights(self):
        """
        Update the weights of the layers by propagating back the error
        """
        # Update the weights layer by layers
        for layer in self.layers:
            layer.updateWeights(self.learning_rate)

    def train(self, verbose=True):
        """Train the Multi-layer Perceptrons

        Parameters
        ----------
        verbose : boolean
            Print logging messages with validation accuracy if verbose is True.
        """
        # Run the training "epochs" times, print out the logs
        for epoch in range(self.epochs):
            if verbose:
                print("Training epoch {0}/{1}.."
                      .format(epoch + 1, self.epochs))

            self._train_one_epoch()

            if verbose:
                accuracy = accuracy_score(self.validation_set.label,
                                          self.evaluate(self.validation_set))
                # Record the performance of each epoch for later usages
                # e.g. plotting, reporting..
                self.performances.append(accuracy)
                print("Accuracy on validation: {0:.2f}%"
                      .format(accuracy * 100))
                print("-----------------------------")

    def _train_one_epoch(self):
        """
        Train one epoch, seeing all input instances
        """

        for img, label in zip(self.training_set.input,
                              self.training_set.label):

            target = np.zeros(10)
            target[label] = 1

            self._feed_forward(img)
            self._compute_error(target)
            self._update_weights()

    def classify(self, test_instance):
        # Classify an instance given the model of the classifier
        self._feed_forward(test_instance)
        return np.argmax(self._get_output_layer().outp)

    def evaluate(self, test=None):
        """Evaluate a whole dataset.

        Parameters
        ----------
        test : the dataset to be classified
        if no test data, the test set associated to the classifier will be used

        Returns
        -------
        List:
            List of classified decisions for the dataset's entries.
        """
        if test is None:
            test = self.test_set.input
        # Once you can classify an instance, just use map for all of the test
        # set.
        return list(map(self.classify, test))

    def __del__(self):
        # Remove the bias from input data
        self.training_set.input = np.delete(self.training_set.input, 0, axis=1)
        self.validation_set.input = np.delete(self.validation_set.input, 0,
                                              axis=1)
        self.test_set.input = np.delete(self.test_set.input, 0, axis=1)
