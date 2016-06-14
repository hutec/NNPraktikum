
from util.activation_functions import Activation
from model.logistic_layer import LogisticLayer
from layer import Layer
from model.classifier import Classifier

from sklearn.metrics import accuracy_score

import numpy as np

import copy

import util.loss_functions as loss_functions

class MultilayerPerceptron(Classifier):
    """
    A multilayer perceptron used for classification
    """

    def __init__(self, train, valid, test, layers=None, input_weights=None,
                 output_task='classification', output_activation='sigmoid',
                 cost='mse', learning_rate=0.01, epochs=50):
        """
        A digit-7 recognizer based on logistic regression algorithm

        Parameters
        ----------
        train : list
        valid : list
        test : list
        layers: list
            List of layers
        input_weights: list
            weight layer
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
        """

        self.learning_rate = learning_rate
        self.epochs = epochs
        self.output_task = output_task  # Either classification or regression
        self.output_activation = output_activation
        self.cost_string = cost
        self.cost = loss_functions.get_loss(cost)

        print("Task: {}, Activation Function {}, Error Function: {}".format(self.output_task,
                                                                            self.output_activation,
                                                                            self.cost_string))

        self.training_set = train
        self.validation_set = valid
        self.test_set = test


        # Record the performance of each epoch for later usages
        # e.g. plotting, reporting..
        self.performances = []

        self.layers = layers
        self.input_weights = input_weights


        if layers is None:
            if output_task == 'classification':
                self.layers = []
                output_activation = "sigmoid"
                self.layers.append(LogisticLayer(train.input.shape[1], 10,
                                                 activation=output_activation,
                                                 is_classifier_layer=False))
                self.layers.append(LogisticLayer(10, 10,
                                                 activation=output_activation,
                                                 is_classifier_layer=False))
                self.layers.append(LogisticLayer(10, 1,
                                                 activation=output_activation,
                                                 is_classifier_layer=True))
            elif output_task == 'classify_all':
                self.layers = []
                self.layers.append(Layer(train.input.shape[1], 10,
                                         activation='softmax',
                                         is_classifier_layer=False))
                self.layers.append(Layer(10, 10,
                                         activation='softmax',
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
        return self.get_layer(0)

    def _get_output_layer(self):
        return self.get_layer(-1)

    def _feed_forward(self, inp):
        """
        Do feed forward through the layers of the network

        Parameters
        ----------
        inp : ndarray
            a numpy array containing the input of the layer

        # Here you have to propagate forward through the layers
        # And remember the activation values of each layer
        """
        for i, layer in enumerate(self.layers):
            if i > 0:
                # Input with value 1 for bias weight
                inp = np.insert(inp, 0, 1, axis=0)
            inp = layer.forward(inp)

        return inp

    def _compute_error(self, target):
        """
        Compute the total error of the network and calculate deltas.

        Returns
        -------
        ndarray :
            a numpy array (1,nOut) containing the output of the layer
        """

        if self.output_task == 'classify_all':
            tmp = np.zeros(10)
            tmp[target] = 1
            target = tmp

        for i, layer in enumerate(reversed(self.layers)):
            if layer.is_classifier_layer:
                next_derivatives = - self.cost.derivative_error(target, layer.outp)
                # next_derivatives = np.array(target - layer.outp)
                next_weights = np.ones(layer.shape[1])

            layer.computeDerivative(next_derivatives, next_weights)
            next_derivatives = layer.deltas
            next_weights = layer.weights[1:]

    def _update_weights(self):
        """
        Update the weights of the layers by propagating back the error
        """
        for layer in self.layers:
            layer.updateWeights(self.learning_rate)

    def train(self, verbose=True):
        """Train the Multi-layer Perceptrons

        Parameters
        ----------
        verbose : boolean
            Print logging messages with validation accuracy if verbose is True.
        """

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

        old_weights = copy.deepcopy(self.layers[0].weights)

        for img, label in zip(self.training_set.input,
                              self.training_set.label):
            self._feed_forward(img)
            self._compute_error(label)
            self._update_weights()


    def classify(self, test_instance):
        # Classify an instance given the model of the classifier
        # You need to implement something here
        outp = self._feed_forward(test_instance)

        if self.output_task == 'classify_all':
            return np.argmax(outp)
        elif self.output_task == 'classify':
            return outp > 0.5
        else:
            print("Unknown task {}".format(self.output_task))
            return False

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
