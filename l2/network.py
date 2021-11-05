import numpy as np

from l2.activation_function import *
from l2.extension_data import x_train_unipolar_aug, y_train_unipolar_aug
from l2.layers import Layer, Softmax, GaussianWeightsInitStrategy

EPOCHS = 10


def print_if_verbose(verbose, to_print):
    if verbose:
        print(to_print)


class Network:
    def __init__(self, input_size, learning_step):
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
        return -1 * np.log(ys_result) * ys_label

    def fit(self, xs, ys, verbose=False):
        epoch_size = np.size(xs)
        for i in range(EPOCHS):
            epoch_error = 0
            loss_bias = [np.zeros(shape=layer_size) for layer_size in
                         map(lambda layer_in_network: np.size(layer_in_network.weights, axis=0), self.layers[::-1])]
            loss = [np.zeros(shape=layer.weights.shape) for layer in self.layers[::-1]]
            softmax_loss_bias = np.zeros(shape=np.size(self.softmax.weights, axis=0))
            softmax_loss = np.zeros(shape=self.softmax.weights.shape)
            for x, y in zip(xs, ys):
                y_predicted = self.predict(x)
                total_loss = self.loss_function(y, y_predicted)
                next_layer_loss = total_loss
                next_layer = self.softmax
                softmax_loss_bias += total_loss
                softmax_loss += np.outer(self.softmax.last_input, total_loss).T
                epoch_error += total_loss.sum()
                for count, layer in enumerate(self.layers[::-1]):
                    derivative = layer.loss_derivative()
                    this_layer_loss = next_layer.weights.T.dot(next_layer_loss) * derivative
                    loss_bias[count] += this_layer_loss
                    loss[count] += np.outer(layer.last_input, this_layer_loss).T
                    next_layer_loss = this_layer_loss
                    next_layer = layer
            for count, layer in enumerate(self.layers[::-1]):
                layer.weights += self.learning_step / epoch_size * loss[count]
                layer.bias += self.learning_step / epoch_size * loss_bias[count].sum()
            self.softmax.weights += self.learning_step / epoch_size * softmax_loss
            self.softmax.bias += self.learning_step / epoch_size * softmax_loss_bias.sum()
            print_if_verbose(verbose, epoch_error / epoch_size)


if __name__ == '__main__':
    x_train = np.array([[1, 1], [2, 2]])

    model = Network(input_size=2, learning_step=0.06)

    model.add_layer(6, act_function=Relu)
    model.add_layer(7, act_function=Tanh)
    model.compile(2)

    print("First prediction")
    print(f"{model.predict(x_train_unipolar_aug[0])}, {y_train_unipolar_aug[0]}")
    print(f"{model.predict(x_train_unipolar_aug[-1])}, {y_train_unipolar_aug[-1]}")
    print("Learning steps:")
    model.fit(x_train_unipolar_aug, y_train_unipolar_aug, verbose=True)
    print("Prediction after learning")
    print(f"{model.predict(x_train_unipolar_aug[0])}, {y_train_unipolar_aug[0]}")
    print(f"{model.predict(x_train_unipolar_aug[-1])}, {y_train_unipolar_aug[-1]}")
