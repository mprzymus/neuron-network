import numpy as np

from l2.activation_function import Relu


class GaussianWeightsInitStrategy:
    def __init__(self, mean=0.0, standard_dev=0.5):
        self.loc = mean
        self.scale = standard_dev

    def init_weights(self, input_size, layer_size):
        return np.random.normal(size=(layer_size, input_size), loc=self.loc, scale=self.scale)


_gaussian = GaussianWeightsInitStrategy()


class Layer:
    def __init__(self, input_size, layer_size, act_function=Relu, weights_init_strategy=_gaussian,
                 bias=None, previous_layer=None):
        self.previous_layer = previous_layer
        self.weights = weights_init_strategy.init_weights(input_size, layer_size)
        self.act_function = act_function
        if bias is None:
            self.bias = weights_init_strategy.init_weights(1, 1)[0]
        else:
            self.bias = bias
        self.last_result = None
        self.last_input = None

    def output_size(self):
        layer_size, _ = self.weights.shape
        return layer_size

    def activate(self, input_vector):
        self.calculate_act_input(input_vector)
        return self.apply_activation()

    def apply_activation(self):
        return np.vectorize(self.act_function.apply)(self.last_result)

    def calculate_act_input(self, input_vector):
        self.last_input = input_vector
        weighted = self.weights.dot(input_vector)
        self.last_result = self.bias + weighted

    def loss_derivative(self):
        return np.vectorize(self.act_function.apply_derivative)(self.last_result)


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
            result = np.exp(input_vector) / np.sum(np.exp(input_vector))
            return result
