import math


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
        try:
            return 1 / (1 + math.exp(-x))
        except OverflowError:
            return 1 if x > 0 else 0

    @staticmethod
    def apply_derivative(sigmoid_result):
        return sigmoid_result * (1 - sigmoid_result)


class Tanh:
    @staticmethod
    def apply(x):
        return 2 / (1 + math.exp(-2 * x)) - 1

    @staticmethod
    def apply_derivative(tanh_result):
        return 1 - tanh_result**2
