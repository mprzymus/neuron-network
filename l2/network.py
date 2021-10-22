import numpy as np

from l2.activation_function import relu, sigmoid, tanh
from l2.layers import Layer, Softmax, GaussianWeightsInitStrategy


class Network:
    def __init__(self, input_size):
        self.layers = []
        self.next_layer_input = input_size
        self.softmax = Softmax()

    def add_layer(self, layer_size, act_function=relu, weights_init_strategy=GaussianWeightsInitStrategy()):
        last_layer_out = self.output_size()
        layer = Layer(last_layer_out, layer_size, act_function=act_function,
                      weights_init_strategy=weights_init_strategy)
        self.layers.append(layer)
        self.next_layer_input = layer_size

    def output_size(self):
        return self.next_layer_input

    def predict(self, xs):
        for layer in self.layers:
            xs = layer.activate(xs)
        return self.softmax.activate(xs)


if __name__ == '__main__':
    x = np.array([4, 1, 3, 4])

    model = Network(4)
    model.add_layer(6, act_function=relu)
    model.add_layer(8, act_function=sigmoid)
    model.add_layer(4, act_function=tanh)
    model.add_layer(2, act_function=relu)

    y = model.predict(x)
    print(y)
