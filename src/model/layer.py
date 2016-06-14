import numpy as np

from util.activation_functions import Activation
import util.loss_functions as loss_functions


class Layer():
    """
    A layer of neural

    Parameters
    ----------
    n_in: int: number of units from the previous layer (or input data)
    n_out: int: number of units of the current layer (or output)
    activation: string: activation function of every units in the layer
    is_classifier_layer: bool:  to do classification or regression

    Attributes
    ----------
    n_in : positive int:
        number of units from the previous layer
    n_out : positive int:
        number of units of the current layer
    weights : ndarray
        weight matrix
    activation : functional
        activation function
    activation_string : string
        the name of the activation function
    is_classifier_layer: bool
        to do classification or regression
    deltas : ndarray
        partial derivatives
    size : positive int
        number of units in the current layer
    shape : tuple
        shape of the layer, is also shape of the weight matrix
    """

    def __init__(self, n_in, n_out, weights=None,
                 activation='sigmoid', is_classifier_layer=False):

        # Get activation function from string
        self.activation_string = activation
        self.activation = Activation.get_activation(self.activation_string)
        self.activation_derivative = Activation.get_derivative(
            self.activation_string)

        self.n_in = n_in
        self.n_out = n_out

        self.inp = np.ndarray((n_in+1, 1))
        self.inp[0] = 1
        self.outp = np.ndarray((n_out, 1))
        self.deltas = np.zeros((n_out, 1))

        # You can have better initialization here
        if weights is None:
            self.weights = np.random.rand(n_in+1, n_out)/10
        else:
            assert(weights.shape == (n_in + 1, n_out))
            self.weights = weights

        self.is_classifier_layer = is_classifier_layer

        # Some handy properties of the layers
        self.size = self.n_out
        self.shape = self.weights.shape

    def forward(self, inp):
        """
        Compute forward step over the input using its weights

        Parameters
        ----------
        inp : ndarray
            a numpy array (n_in + 1,1) containing the input of the layer

        Change outp
        -------
        outp: ndarray
            a numpy array (n_out,1) containing the output of the layer
        """
        self.inp = inp
        outp = self._fire(inp)
        self.outp = outp

        return outp

    def computeDerivative(self, next_derivatives, next_weights):
        """
        Compute the derivatives (backward)

        Parameters
        ----------
        next_derivatives: ndarray
            a numpy array containing the derivatives from next layer
        next_weights : ndarray
            a numpy array containing the weights from next layer

        Change deltas
        -------
        deltas: ndarray
            a numpy array containing the partial derivatives on this layer
        """

        if self.is_classifier_layer:
            self.deltas = self.activation_derivative(self.outp) * next_derivatives
        else:
            self.deltas = (self.activation_derivative(self.outp) *
                           np.tensordot(next_derivatives, next_weights, axes=([0], [-1])))

    def updateWeights(self, learning_rate):
        """
        Update the weights of the layer
        """

        for neuron in range(0, self.n_out):
            self.weights[:, neuron] += (learning_rate *
                                        self.deltas[neuron] *
                                        self.inp)

    def _fire(self, inp):
        return self.activation(np.dot(np.array(inp), self.weights))
