# -*- coding: utf-8 -*-
import sys
import logging
import numpy as np
from model.logistic_layer import LogisticLayer
from model.auto_encoder import AutoEncoder

from copy import deepcopy
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score

from util.loss_functions import MeanSquaredError, AbsoluteError

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.DEBUG,
                    stream=sys.stdout)


class DenoisingAutoEncoder(AutoEncoder):
    """
    A denoising autoencoder.
    Since the input are greyscale images sigmoid activation function and
    cross entropy error ist used.
    """

    def __init__(self, train, valid, test, learning_rate=0.1, epochs=30, hidden_units=100, corruption=0.2):
        """
         Parameters
        ----------
        train : list
        valid : list
        test : list
        learning_rate : float
        epochs : positive int
        hidden_layers : int
        corruption: float

        Attributes
        ----------
        training_set : list
        validation_set : list
        test_set : list
        learning_rate : float
        epochs : positive int
        performances: array of floats
        """
        self.learning_rate = learning_rate
        self.epochs = epochs

        self.training_set = train
        self.validation_set = valid
        self.test_set = test

        self.number_of_hidden_units = hidden_units
        self.corruption = corruption

        self.error = MeanSquaredError()

        # Eventually add those as parameters
        self.layers = None
        self.input_weights = None

        self.performances = []

        if self.layers is None:
            self.layers = []

            # Hidden Layer
            self.layers.append(LogisticLayer(train.input.shape[1],
                                             self.number_of_hidden_units, None,
                                             activation="sigmoid",
                                             is_classifier_layer=False))

            # Output layer
            self.layers.append(LogisticLayer(self.number_of_hidden_units,
                                             train.input.shape[1], None,
                                             activation="sigmoid",
                                             is_classifier_layer=True))

        # add bias values ("1"s) at the beginning of all data sets
        self.training_set.input = np.insert(self.training_set.input, 0, 1,
                                            axis=1)
        self.validation_set.input = np.insert(self.validation_set.input, 0, 1,
                                              axis=1)
        self.test_set.input = np.insert(self.test_set.input, 0, 1, axis=1)

    def train(self, verbose=True):
        """
        Train the denoising autoencoder
        """

        for epoch in range(self.epochs):
            if verbose:
                print("Training epoch {0}/{1}.."
                      .format(epoch + 1, self.epochs))

            self._train_one_epoch()

            if verbose:
                accuracy = self.evaluate(self.validation_set)
                accuracy = np.max(accuracy)


                self.performances.append(accuracy)
                print("Accuracy on validation: {0:.3f}%"
                      .format(accuracy * 100))
                print("-----------------------------")

    def calculate_error(self, test_instance):
        # Classify an instance given the model of the classifier
        self._feed_forward(test_instance)
        return self.error.calculate_error(np.delete(test_instance, 0), self._get_output_layer().outp)

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
        return list(map(self.calculate_error, test))

    def save_weights(self):
        np.save("weights", self._get_weights())

    def save_features(self):
        weights = self._get_weights()
        weights = np.delete(weights, 0 , axis=0)
        for i in range(weights.shape[1]):
            plt.imsave("features/feature_{}.png".format(i), weights[:, i].reshape(28, 28), cmap='gray')



    def _train_one_epoch(self):
        """
        Train one epoch, seeing all input instances
        """

        for img, label in zip(self.training_set.input,
                              self.training_set.label):

            target = deepcopy(img)
            target = np.delete(target, 0)
            # r = int(target.shape[0] * self.corruption)
            # mask = np.random.randint(0, target.shape[0], size=r)
            # target[mask] = 0

            self._feed_forward(img)
            self._compute_error(target)
            self._update_weights()

    def _get_weights(self):
        """
        Get the weights (after training)
        """
        return self._get_layer(0).weights

    def _feed_forward(self, inp):
        """
        Do feed forward through the layers of the network

        """
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

    def _get_output_layer(self):
        return self._get_layer(-1)

    def _get_layer(self, layer_index):
        return self.layers[layer_index]


    def __del__(self):
        # Remove the bias from input data
        self.training_set.input = np.delete(self.training_set.input, 0, axis=1)
        self.validation_set.input = np.delete(self.validation_set.input, 0,
                                              axis=1)
        self.test_set.input = np.delete(self.test_set.input, 0, axis=1)
