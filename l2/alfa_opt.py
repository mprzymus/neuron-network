from math import sqrt

import numpy as np

EPS = 1e-2


class NoOptimizer:
    def init_optimizer(self, model):
        pass

    def apply_optimizer(self, gradient, layer_number):
        return gradient

    def apply_optimizer_bias(self, gradient, layer_number):
        return gradient


class Adagrad(NoOptimizer):
    def __init__(self):
        self.previous_grad_weights = []
        self.previous_grad_biases = []

    def init_optimizer(self, model):
        for layer in model.layers[::-1]:
            self.previous_grad_weights.append(np.zeros(np.shape(layer.weights)))
            self.previous_grad_biases.append(np.zeros(np.shape(layer.bias)))
        self.previous_grad_weights.append(np.zeros(np.shape(model.softmax.weights)))
        self.previous_grad_biases.append(np.zeros(np.shape(model.softmax.bias)))

    def apply_optimizer(self, gradient, layer_number):
        change = np.sqrt(self.previous_grad_weights[layer_number] + EPS)
        self.previous_grad_weights[layer_number] += gradient ** 2
        return gradient / change

    def apply_optimizer_bias(self, gradient, layer_number):
        change = np.sqrt(self.previous_grad_biases[layer_number] + EPS)
        self.previous_grad_biases[layer_number] += gradient ** 2
        return gradient / change
