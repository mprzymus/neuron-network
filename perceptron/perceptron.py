import numpy as np

from activation_function import threshold_unipolar


class Perceptron:
    def __init__(self, weights=None, act_function=threshold_unipolar, alfa=0.01, size=2, bias=None, bias_static=False,
                 max_weight=1):
        self.weights = weights
        self.bias_static = bias_static
        self.epochs = 0
        self.alfa = alfa
        self.act_function = act_function
        if bias is None:
            self.bias = np.random.rand() * 2 - 1
        else:
            self.bias = bias
        self.size = size
        if self.weights is None:
            self.weights = (np.random.rand(size) * 2 - 1) * max_weight
            self.weights[1] *= 1

    def predict(self, x):
        x_s = x * self.weights
        total = np.sum(x_s) + self.bias
        return self.act_function(total)

    def fit(self, x_train, y_train, verbose=False):
        was_error = True
        i = 0
        while was_error:
            was_error = False
            if verbose:
                print(f"epoch {i}")
                print(f"{self.weights}, {self.bias}")
            i += 1
            for count, x in enumerate(x_train):
                y_predict = self.predict(x)
                delta = y_train[count] - y_predict
                self.weights += delta * self.alfa * x
                if not self.bias_static:
                    self.bias += delta * self.alfa
                was_error = was_error if was_error else y_predict != y_train[count]
        self.epochs = i
