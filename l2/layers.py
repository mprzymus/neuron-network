import numpy as np

from l2.activation_function import Relu
from l2.weight_init_factory import GaussianWeightsInitStrategy, WeightInitStrategy

_gaussian = GaussianWeightsInitStrategy()


class Layer:
    def __init__(self, input_size, layer_size, act_function=Relu, weights_init_strategy: WeightInitStrategy = _gaussian,
                 bias=None, previous_layer=None):
        self.previous_layer = previous_layer
        self.weights = weights_init_strategy.init_weights(input_size, layer_size)
        self.act_function = act_function
        if bias is None:
            self.bias = weights_init_strategy.init_weights(layer_size, 1)[0]
            self.bias = np.positive(self.bias)
        else:
            self.bias = bias
        self.last_result = None
        self.last_input = None
        self.last_act = None

    def output_size(self):
        layer_size, _ = self.weights.shape
        return layer_size

    def activate(self, input_vector):
        self.calculate_act_input(input_vector)
        return self.apply_activation()

    def apply_activation(self):
        self.last_act = np.vectorize(self.act_function.apply)(self.last_result)
        return self.last_act

    def calculate_act_input(self, input_vector):
        self.last_input = input_vector
        weighted = self.weights.dot(input_vector)
        self.last_result = weighted + self.bias

    def last_act_derivative(self):
        return np.vectorize(self.act_function.apply_derivative)(self.last_act)


class Softmax(Layer):
    def __init__(self, input_size, layer_size, weights_init_strategy=_gaussian, bias=None):
        super().__init__(input_size, layer_size, weights_init_strategy=weights_init_strategy, bias=bias)
        self.act_function = self.SoftmaxFun

    def activate(self, input_vector):
        self.calculate_act_input(input_vector)
        return self.act_function.apply(self.last_result)

    class SoftmaxFun:
        @staticmethod
        def apply(input_vector):
            subtract = input_vector - np.max(input_vector)
            e = np.exp(subtract)
            if e.ndim == 1:
                return e / np.sum(e, axis=0)
            else:
                return e / np.sum(e, axis=1, keepdims=True)
