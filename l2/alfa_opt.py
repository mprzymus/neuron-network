from math import sqrt

import numpy as np

EPS = 1e-8


class OptimizerStrategy:
    def init_optimizer(self, model):
        pass

    def apply_optimizer(self, gradient, alfa, layer_number):
        return gradient

    def apply_optimizer_bias(self, gradient, alfa, layer_number):
        return gradient


class Adagrad(OptimizerStrategy):
    def __init__(self):
        self.previous_grad_weights = []
        self.previous_grad_biases = []

    def init_optimizer(self, model):
        for layer in model.layers[::-1]:
            self.previous_grad_weights.append(np.zeros(np.shape(layer.weights)))
            self.previous_grad_biases.append(np.zeros(np.shape(layer.bias)))
        self.previous_grad_weights.append(np.zeros(np.shape(model.softmax.weights)))
        self.previous_grad_biases.append(np.zeros(np.shape(model.softmax.bias)))

    def apply_optimizer(self, gradient, alfa, layer_number):
        change = self.count_rms(layer_number)
        self.update_grad_weights(gradient, layer_number)
        return gradient / change * alfa

    def update_grad_weights(self, gradient, layer_number):
        self.previous_grad_weights[layer_number] += gradient ** 2

    def count_rms(self, layer_number):
        return np.sqrt(self.previous_grad_weights[layer_number] + EPS)

    def apply_optimizer_bias(self, gradient, alfa, layer_number):
        change = self.count_rms_bias(layer_number)
        self.update_grad_weights_bias(gradient, layer_number)
        return gradient / change * alfa

    def update_grad_weights_bias(self, gradient, layer_number):
        self.previous_grad_biases[layer_number] += gradient ** 2

    def count_rms_bias(self, layer_number):
        return np.sqrt(self.previous_grad_biases[layer_number] + EPS)


class Adadelta(OptimizerStrategy):
    def __init__(self, gamma=0.9):
        self.gamma = gamma
        self.delta_weight_rms = []
        self.delta_bias_rms = []
        self.previous_grad_weights = []
        self.previous_grad_biases = []

    def init_optimizer(self, model):
        for layer in model.layers[::-1]:
            self.delta_weight_rms.append(np.zeros(np.shape(layer.weights)))
            self.delta_bias_rms.append(np.zeros(np.shape(layer.bias)))
            self.previous_grad_weights.append(np.zeros(np.shape(layer.weights)))
            self.previous_grad_biases.append(np.zeros(np.shape(layer.bias)))
        self.delta_weight_rms.append(np.zeros(np.shape(model.softmax.weights)))
        self.delta_bias_rms.append(np.zeros(np.shape(model.softmax.bias)))
        self.previous_grad_weights.append(np.zeros(np.shape(model.softmax.weights)))
        self.previous_grad_biases.append(np.zeros(np.shape(model.softmax.bias)))

    def apply_optimizer(self, gradient, alfa, layer_number):
        result, self.previous_grad_weights[layer_number], self.delta_weight_rms[layer_number] = self.method_name(
            gradient, self.previous_grad_weights[layer_number], self.delta_weight_rms[layer_number]
        )
        return result

    def method_name(self, gradient, previous_grad, delta_weight):
        gradient_rms = np.sqrt(previous_grad + EPS)
        delta_grad = self.gamma * previous_grad + (1 - self.gamma) * gradient ** 2
        previous_grad = np.sqrt(delta_grad + EPS)
        result = delta_weight * gradient / gradient_rms
        delta_weight = self.gamma * delta_weight + (1 - self.gamma) * (result ** 2)
        delta_weight = np.sqrt(delta_weight + EPS)
        return result, previous_grad, delta_weight

    def apply_optimizer_bias(self, gradient, alfa, layer_number):
        result, self.previous_grad_biases[layer_number], self.delta_bias_rms[layer_number] = self.method_name(
            gradient, self.previous_grad_biases[layer_number], self.delta_bias_rms[layer_number]
        )
        return result


class Adam(OptimizerStrategy):
    def __init__(self, gradient_factor=0.9, square_gradient_factor=0.999):
        self.gradient_factor = gradient_factor
        self.square_gradient_factor = square_gradient_factor
        self.previous_gradient_weight = []
        self.previous_gradient_bias = []
        self.previous_gradient_squared_weight = []
        self.previous_gradient_squared_bias = []
        self.number_of_iterations = 1

    def init_optimizer(self, model):
        for layer in model.layers[::-1]:
            self.previous_gradient_squared_weight.append(np.zeros(np.shape(layer.weights)))
            self.previous_gradient_weight.append(np.zeros(np.shape(layer.weights)))
            self.previous_gradient_bias.append(np.zeros(np.shape(layer.bias)))
            self.previous_gradient_squared_bias.append(np.zeros(np.shape(layer.bias)))
        self.previous_gradient_squared_weight.append(np.zeros(np.shape(model.softmax.weights)))
        self.previous_gradient_weight.append(np.zeros(np.shape(model.softmax.weights)))
        self.previous_gradient_bias.append(np.zeros(np.shape(model.softmax.bias)))
        self.previous_gradient_squared_bias.append(np.zeros(np.shape(model.softmax.bias)))

    def apply_optimizer(self, gradient, alfa, layer_number):
        self.previous_gradient_weight[layer_number] = self.gradient_factor * self.previous_gradient_weight[
            layer_number] + (1 - self.gradient_factor) * gradient
        self.previous_gradient_squared_weight[
            layer_number] = self.square_gradient_factor * self.previous_gradient_squared_weight[layer_number] + (
                1 - self.square_gradient_factor) * (gradient ** 2)
        corrected_gradient = self.previous_gradient_weight[layer_number] / (
                1 - np.power(self.gradient_factor, self.number_of_iterations))
        corrected_gradient_square = self.previous_gradient_squared_weight[layer_number] / (
                1 - np.power(self.square_gradient_factor, self.number_of_iterations))
        return alfa * corrected_gradient / (np.sqrt(corrected_gradient_square) + EPS)

    def apply_optimizer_bias(self, gradient, alfa, layer_number):
        return super().apply_optimizer_bias(gradient, alfa, layer_number)
