import time

import numpy as np

from l2.activation_function import relu
from l2.layers import Layer, Softmax, GaussianWeightsInit


class Network:
    def __init__(self, first_layer):
        self.layers = [first_layer]

    def add_layer(self, layer_size, act_function=relu, weights_init_strategy=GaussianWeightsInit()):
        last_layer_out = self.output_size()
        layer = Layer(last_layer_out, layer_size, act_function=act_function,
                      weights_init_strategy=weights_init_strategy)
        self.layers.append(layer)

    def output_size(self):
        return self.layers[-1].output_size()

    def setup(self):
        self.layers.append(Softmax())

    def predict(self, xs):
        for layer in self.layers:
            xs = layer.activate(xs)
        return xs


if __name__ == '__main__':
    l1 = Layer(4, 6)
    model = Network(l1)
    model.add_layer(8)
    model.add_layer(3)
    model.setup()

    x = np.arange(4)

    y = model.predict(x)
    print(y)
