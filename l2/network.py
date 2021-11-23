import datetime

import numpy as np
from tensorflow import keras

from l2.activation_function import *
from l2.extension_data import x_train_unipolar_aug, y_train_unipolar_aug
from l2.layers import Layer, Softmax, GaussianWeightsInitStrategy


def print_if_verbose(verbose, to_print):
    if verbose:
        print(f"{datetime.datetime.now()}: {to_print}")


class Network:
    def __init__(self, input_size, learning_step=0.01, gradient_clip=10000, min_epochs=10):
        self.layers = []
        self.last_layer = None
        self.next_layer_input = input_size
        self.softmax = None
        self.learning_step = learning_step
        self.gradient_clip = gradient_clip
        self.min_epochs = min_epochs

    def add_layer(self, layer_size, bias=None, act_function=Relu, weights_init_strategy=GaussianWeightsInitStrategy()):
        last_layer_out = self.output_size()
        layer = Layer(last_layer_out, layer_size, act_function=act_function, bias=bias,
                      weights_init_strategy=weights_init_strategy, previous_layer=self.last_layer)
        self.last_layer = layer
        self.layers.append(layer)
        self.next_layer_input = layer_size

    def compile(self, number_of_classes, bias=None, weights_init_strategy=GaussianWeightsInitStrategy()):
        self.softmax = Softmax(self.next_layer_input, number_of_classes, weights_init_strategy=weights_init_strategy,
                               bias=bias)
        self.softmax.previous_layer = self.layers[-1]

    def output_size(self):
        return self.next_layer_input

    def predict(self, xs):
        for layer in self.layers:
            xs = layer.activate(xs)
        return self.softmax.activate(xs)

    def loss_function(self, ideal, actual):
        return np.log(1e-30 + actual) * ideal * -1.0

    def predict_all(self, xs):
        results = []
        for x in xs:
            results.append(self.predict(x))
        return results

    def fit(self, xs, ys, x_valid, y_valid, verbose=False, target_loss=0.5, batch_size=None, max_epochs=7):
        valid_errors = []
        train_errors = []
        epoch_size = len(xs)
        batch_size = epoch_size if batch_size is None else batch_size
        epochs_counter = 0
        valid_error = self.count_loss_on_data(x_valid, y_valid)
        print_if_verbose(verbose, f"valid_loss {valid_error}")
        while valid_error > target_loss and epochs_counter < max_epochs:
            start_batch_from = 0
            epochs_counter += 1
            epoch_error = 0
            while start_batch_from < epoch_size:
                batch_x = xs[start_batch_from:start_batch_from + batch_size]
                epoch_error = self.perform_batch(epoch_error, batch_x,
                                                 ys[start_batch_from:start_batch_from + batch_size])
                start_batch_from += batch_size
            valid_error = self.count_loss_on_data(x_valid, y_valid)
            train_error = self.count_loss_on_data(xs, ys)
            print_if_verbose(verbose, f"learn_loss: {train_error}, valid_loss {valid_error}")
            valid_errors.append(valid_error)
            train_errors.append(train_error)
            if self.is_over_fitting(valid_errors):
                break
        return train_errors, valid_errors

    def perform_batch(self, epoch_error, xs, ys):
        batch_size = len(xs)
        error_bias = self.init_loss_bias()
        error = self.init_loss()
        softmax_error_bias = self.init_softmax_loss_bias()
        softmax_error = self.init_softmax_loss()
        for x, y in zip(xs, ys):
            y_predicted = self.predict(x)
            predict_loss = self.loss_function(y, y_predicted)
            next_layer = self.softmax
            predict_loss_der = y_predicted - y
            next_layer_error = predict_loss_der
            softmax_error_bias += predict_loss_der
            softmax_error += np.outer(predict_loss_der, self.softmax.last_input)
            epoch_error += predict_loss.sum()
            #  spróbować od nowa może
            for layer_number, layer in enumerate(self.layers[::-1]):
                derivative = layer.last_act_derivative()
                this_layer_error = next_layer.weights.T.dot(next_layer_error) * derivative
                error_bias[layer_number] += this_layer_error
                error[layer_number] += np.outer(layer.last_input, this_layer_error).T
                next_layer_error = this_layer_error
                next_layer = layer
        self.update_weights(batch_size, error, error_bias, softmax_error, softmax_error_bias)
        return epoch_error

    def update_weights(self, batch_size, loss, loss_bias, softmax_loss, softmax_loss_bias):
        # print("zmiany wag")
        for layer_number, layer in enumerate(self.layers[::-1]):
            weights = self.clip_gradient(self.learning_step / batch_size * loss[layer_number])
            # print(weights)
            layer.weights -= weights
            bias = self.clip_gradient(self.learning_step / batch_size * loss_bias[layer_number])
            # print(bias)
            layer.bias -= bias
        last_layer_weights = self.clip_gradient(self.learning_step / batch_size * softmax_loss)
        self.softmax.weights -= last_layer_weights
        # print(last_layer_weights)

        last_layer_bias = self.clip_gradient(self.learning_step / batch_size * softmax_loss_bias)
        self.softmax.bias -= last_layer_bias
        #  print(last_layer_bias)

    def clip_gradient(self, gradient):
        return np.where(np.abs(gradient) < self.gradient_clip, gradient, np.sign(gradient) * self.gradient_clip)

    def init_softmax_loss(self):
        return np.zeros(shape=self.softmax.weights.shape)

    def init_softmax_loss_bias(self):
        return np.zeros(shape=np.size(self.softmax.weights, axis=0))

    def init_loss(self):
        return [np.zeros(shape=layer.weights.shape) for layer in self.layers[::-1]]

    def init_loss_bias(self):
        return [np.zeros(shape=layer_size) for layer_size in
                map(lambda layer_in_network: np.size(layer_in_network.weights, axis=0), self.layers[::-1])]

    def count_loss_on_data(self, x_valid, y_valid):
        predictions = self.predict_all(x_valid)
        valid_error = self.count_valid_error(predictions, y_valid)
        return valid_error

    def count_valid_error(self, predictions, y_valid):
        error = 0
        valid_size = len(y_valid)
        for y_predict, y_actual in zip(predictions, y_valid):
            error += np.sum(self.loss_function(y_actual, y_predict)) / valid_size
        return error

    def is_over_fitting(self, valid_errors):
        if len(valid_errors) >= self.min_epochs:
            return valid_errors[-1] > valid_errors[-2] > valid_errors[-3]
        else:
            return False


'''
    zdecydowanie sposób zmiany wag
    rosną zdecydowanie
'''
