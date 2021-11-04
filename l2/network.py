import numpy as np

from l2.activation_function import *
from l2.layers import Layer, Softmax, GaussianWeightsInitStrategy


class Network:
    def __init__(self, input_size, learning_step):
        self.layers = []
        self.next_layer_input = input_size
        self.softmax = None
        self.learning_step = learning_step

    def add_layer(self, layer_size, act_function=Relu, weights_init_strategy=GaussianWeightsInitStrategy()):
        last_layer_out = self.output_size()
        layer = Layer(last_layer_out, layer_size, act_function=act_function,
                      weights_init_strategy=weights_init_strategy)
        self.layers.append(layer)
        self.next_layer_input = layer_size

    def compile(self, number_of_classes):
        self.softmax = Softmax(self.next_layer_input, number_of_classes)

    def output_size(self):
        return self.next_layer_input

    def predict(self, xs):
        for layer in self.layers:
            xs = layer.activate(xs)
        return self.softmax.activate(xs)

    def loss_function(self, ys_label, ys_result):
        return -np.log(ys_result) * ys_label

    def fit(self, xs, ys):
        for x, y in zip(xs[::-1], ys[::-1]):
            y_predicted = self.predict(x)
            loss = np.zeros(shape=np.size(self.layers) + 1)
            total_loss = self.loss_function(y, y_predicted)
            next_layer_loss = total_loss
            prev_layer = self.softmax
            print(total_loss.sum())
            for count, layer in enumerate(self.layers[::-1]):
                derivative = layer.loss_derivative()
                this_layer_loss = prev_layer.weights.T.dot(next_layer_loss)
                this_layer_loss *= derivative
                next_layer_loss = this_layer_loss
                prev_layer = layer
                print(this_layer_loss)
        return "done"


if __name__ == '__main__':
    x_train = np.array([[1, 1], [2, 2]])

    model = Network(2, 0.01)
    model.add_layer(6, act_function=Relu)
    model.add_layer(8, act_function=Sigmoid)
    model.add_layer(4, act_function=Relu)
    model.compile(2)

    y_pred = model.predict(x_train[0])
    print(y_pred)

    print(model.fit(x_train, np.array([[0, 1]])))
