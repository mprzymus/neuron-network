import numpy as np

from l2.activation_function import relu


class GaussianWeightsInit:
    def __init__(self, mean=0.0, standard_dev=1.0):
        self.loc = mean
        self.scale = standard_dev

    def init_weights(self, input_size, layer_size):
        return np.random.normal(size=(input_size, layer_size), loc=self.loc, scale=self.scale)


class Layer:
    def __init__(self, input_size, layer_size, act_function=relu, weights_init_strategy=GaussianWeightsInit()):
        self.weights = weights_init_strategy.init_weights(input_size, layer_size)
        self.act_function = act_function

    def activate(self, input_vector):
        weighted = self.weights.dot(input_vector)
        return np.vectorize(self.act_function)(weighted)


class Softmax:
    @staticmethod
    def activate(input_vector):
        exp_sum = np.exp(input_vector) / np.sum(np.exp(input_vector))
        return exp_sum
