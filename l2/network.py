import datetime

from tensorflow import keras

from l2.activation_function import *
from l2.extension_data import x_train_unipolar_aug, y_train_unipolar_aug
from l2.layers import Layer, Softmax, GaussianWeightsInitStrategy


def print_if_verbose(verbose, to_print):
    if verbose:
        print(f"{datetime.datetime.now()}: {to_print}")


class Network:
    def __init__(self, input_size, learning_step=0.01):
        self.layers = []
        self.last_layer = None
        self.next_layer_input = input_size
        self.softmax = None
        self.learning_step = learning_step

    def add_layer(self, layer_size, act_function=Relu, weights_init_strategy=GaussianWeightsInitStrategy()):
        last_layer_out = self.output_size()
        layer = Layer(last_layer_out, layer_size, act_function=act_function,
                      weights_init_strategy=weights_init_strategy, previous_layer=self.last_layer)
        self.last_layer = layer
        self.layers.append(layer)
        self.next_layer_input = layer_size

    def compile(self, number_of_classes):
        self.softmax = Softmax(self.next_layer_input, number_of_classes)
        self.softmax.previous_layer = self.layers[-1]

    def output_size(self):
        return self.next_layer_input

    def predict(self, xs):
        for layer in self.layers:
            xs = layer.activate(xs)
        return self.softmax.activate(xs)

    def loss_function(self, ys_label, ys_result):
        return np.log(1e-15 + ys_result) * ys_label * -1.0

    def predict_all(self, xs):
        results = []
        for x in xs:
            results.append(self.predict(x))
        return results

    def fit(self, xs, ys, x_valid, y_valid, verbose=False, wanted_error=0.5, batch_size=None, max_epochs=7):
        epoch_size = len(xs)
        batch_size = epoch_size if batch_size is None else batch_size
        epochs_counter = 0
        valid_error = self.count_loss_on_data(x_valid, y_valid)
        print_if_verbose(verbose, f"valid_loss {valid_error}")
        while valid_error > wanted_error and epochs_counter < max_epochs:
            start_batch_from = 0
            epochs_counter += 1
            epoch_error = 0
            while start_batch_from < epoch_size:
                batch_x = xs[start_batch_from:start_batch_from + batch_size]
                epoch_error = self.perform_batch(epoch_error, batch_x,
                                                 ys[start_batch_from:start_batch_from + batch_size])
                start_batch_from += batch_size
            valid_error = self.count_loss_on_data(x_valid, y_valid)
            print_if_verbose(verbose, f"learn_loss: {epoch_error / epoch_size}, valid_loss {valid_error}")

    def perform_batch(self, epoch_error, xs, ys):
        batch_size = len(xs)
        loss_bias = self.init_loss_bias()
        loss = self.init_loss()
        softmax_loss_bias = self.init_softmax_loss_bias()
        softmax_loss = self.init_softmax_loss()
        for x, y in zip(xs, ys):
            y_predicted = self.predict(x)
            predict_loss = self.loss_function(y, y_predicted)
            next_layer_loss = predict_loss
            next_layer = self.softmax
            predict_loss_der = -(y - y_predicted)
            softmax_loss_bias += predict_loss_der
            softmax_loss += np.outer(self.softmax.last_input, predict_loss_der).T
            epoch_error += predict_loss.sum()
            for layer_number, layer in enumerate(self.layers[::-1]):
                derivative = layer.last_act_derivative()
                this_layer_loss = next_layer.weights.T.dot(next_layer_loss) * derivative
                loss_bias[layer_number] += this_layer_loss
                loss[layer_number] += np.outer(layer.last_input, this_layer_loss).T  # tego jeszcze nie ma
                next_layer_loss = this_layer_loss
                next_layer = layer
        self.update_weights(batch_size, loss, loss_bias, softmax_loss, softmax_loss_bias)
        return epoch_error

    def update_weights(self, batch_size, loss, loss_bias, softmax_loss, softmax_loss_bias):
        for layer_number, layer in enumerate(self.layers[::-1]):
            layer.weights -= self.learning_step / batch_size * loss[layer_number]
            layer.bias -= self.learning_step / batch_size * loss_bias[layer_number].sum()
        self.softmax.weights -= self.learning_step / batch_size * softmax_loss
        self.softmax.bias -= self.learning_step / batch_size * softmax_loss_bias.sum()

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
