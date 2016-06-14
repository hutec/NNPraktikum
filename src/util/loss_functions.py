# -*- coding: utf-8 -*-


"""
Loss functions.
"""

import numpy as np

from abc import ABCMeta, abstractmethod, abstractproperty


class Error:
    """
    Abstract class of an Error
    """
    __metaclass__ = ABCMeta

    @abstractproperty
    def error_string(self):
        pass

    @abstractmethod
    def calculate_error(self, target, output):
        # calculate the error between target and output
        pass

    def derivative_error(self, target, output):
        # derivative of loss function
        pass


class AbsoluteError(Error):
    """
    The Loss calculated by the number of differences between target and output
    """
    def error_string(self):
        self.error_string = 'absolute'

    def calculate_error(self, target, output):
        # It is the numbers of differences between target and output
        return abs(target - output)


class DifferentError(Error):
    """
    The Loss calculated by the number of differences between target and output
    """
    def error_string(self):
        self.error_string = 'different'

    def calculate_error(self, target, output):
        # It is the numbers of differences between target and output
        return target - output


class MeanSquaredError(Error):
    """
    The Loss calculated by the mean of the total squares of differences between
    target and output.
    """
    def error_string(self):
        self.error_string = 'mse'

    def calculate_error(self, target, output):
        # MSE = 1/n*sum (i=1 to n) of (target_i - output_i)^2)
        return (1/len(target))*np.sum((target-output)**2)

    def derivative_error(self, target, output):
        return -(target - output)


class SumSquaredError(Error):
    """
    The Loss calculated by the sum of the total squares of differences between
    target and output.
    """
    def error_string(self):
        self.error_string = 'sse'

    def calculate_error(self, target, output):
        # SSE = 1/2*sum (i=1 to n) of (target_i - output_i)^2)
        return 0.5*np.sum((target-output)**2)


class BinaryCrossEntropyError(Error):
    """
    The Loss calculated by the Cross Entropy between binary target and
    probabilistic output (BCE)
    """
    def error_string(self):
        self.error_string = 'bce'

    def calculate_error(self, target, output):
        # Here you have to implement the Binary Cross Entropy
        return np.sum(target * np.log(output) + (1 - target) * np.log(1 - output))


class CrossEntropyError(Error):
    """
    The Loss calculated by the more general Cross Entropy between two
    probabilistic distributions.
    """
    def error_string(self):
        self.error_string = 'crossentropy'

    def calculate_error(self, target, output):
        # Here you have to implement the Cross Entropy Error
        return np.sum(target * np.log(output) + (1 - target) * np.log(1 - output))

    def derivative_error(self, target, output):
        return  -(target / output) + (1 - target) / (1 - output)


def get_loss(function_name):
    """
    Returns the loss function corresponding to the given string
    """

    if function_name == 'mse':
        return MeanSquaredError()
    elif function_name == 'crossentropy':
        return CrossEntropyError()
    else:
        raise ValueError('Unknown loss function: ' + function_name)

