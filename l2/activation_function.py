import math

import numpy as np


class Relu:
    @staticmethod
    def apply(x):
        return 0 if x < 0 else x

    @staticmethod
    def apply_derivative(relu_result):
        if relu_result == 0:
            return 0.5
        return 0 if relu_result < 0 else 1


class Sigmoid:
    @staticmethod
    def apply(x):
        if x < 0:
            return 1 - 1 / (1 + math.exp(x))
        else:
            return 1 / (1 + math.exp(-x))

    @staticmethod
    def apply_derivative(sigmoid_result):
        return sigmoid_result * (1.0 - sigmoid_result)


class Tanh:
    @staticmethod
    def apply(x):
        return np.tanh(x)

    @staticmethod
    def apply_derivative(tanh_result):
        return 1 - tanh_result**2
