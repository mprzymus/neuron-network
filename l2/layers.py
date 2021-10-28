import numpy as np

from l2.activation_function import relu


class GaussianWeightsInitStrategy:
    def __init__(self, mean=0.0, standard_dev=1.0):
        self.loc = mean
        self.scale = standard_dev

    def init_weights(self, input_size, layer_size):
        return np.random.normal(size=(layer_size, input_size), loc=self.loc, scale=self.scale)


_gaussian = GaussianWeightsInitStrategy()


class Layer:
    def __init__(self, input_size, layer_size, act_function=relu, weights_init_strategy=_gaussian,
                 bias=None):
        self.weights = weights_init_strategy.init_weights(input_size, layer_size)
        self.act_function = act_function
        if bias is None:
            self.bias = weights_init_strategy.init_weights(1, 1)[0]
        else:
            self.bias = bias
        self.last_result = None

    def output_size(self):
        layer_size, _ = self.weights.shape
        return layer_size

    def activate(self, input_vector):
        weighted = self.weights.dot(input_vector)
        weighted += self.bias
        self.last_result = np.vectorize(self.act_function)(weighted)
        return self.last_result


class Softmax:
    def activate(self, input_vector):
        result = np.exp(input_vector) / np.sum(np.exp(input_vector))
        return result

    def output_size(self):
        pass
