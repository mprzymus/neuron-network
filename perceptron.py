import numpy as np


def threshold_bipolar(value):
    return 1 if value > 0 else -1


def threshold_unipolar(value):
    return 1 if value > 0 else 0


class Perceptron:
    def __init__(self, weights=None, act_function=threshold_unipolar, alfa=0.01, size=2):
        self.weights = weights
        self.act_function = act_function
        self.bias = np.random.rand() * 2 - 1
        self.alfa = alfa
        self.size = size
        if self.weights is None:
            self.weights = np.random.rand(size) * 2 - 1
            self.weights[1] *= 1

    def predict(self, x):
        x_s = x * self.weights
        total = np.sum(x_s) + self.bias
        return self.act_function(total)

    def fit(self, x_train, y_train):
        was_error = True
        i = 0
        while was_error:
            was_error = False
            print(f"epoch {i}")
            print(f"{self.weights}, {self.bias}")
            i += 1
            for count, x in enumerate(x_train):
                y_predict = self.predict(x)
                delta = y_train[count] - y_predict
                self.weights += delta * self.alfa * x
                self.bias += delta * self.alfa
                was_error = was_error if was_error else y_predict != y_train[count]
