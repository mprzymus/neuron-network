import numpy as np

from activation_function import threshold_bipolar


class Adaline:
    def __init__(self, weights=None, act_function=threshold_bipolar, alfa=0.01, size=2):
        self.weights = weights
        self.act_function = act_function
        self.bias = np.random.rand() * 2 - 1
        self.alfa = alfa
        self.size = size
        if self.weights is None:
            self.weights = np.random.rand(size) * 2 - 1
            self.weights[1] *= 1

    def predict(self, x, verbose=False):
        total = self.sum_input(x)
        if verbose:
            print(total)
        return self.act_function(total)

    def sum_input(self, x):
        return np.dot(x, self.weights) + self.bias

    def fit(self, x_train, y_train, stop_error=0.2406):
        set_size = np.size(x_train)
        error_avg = stop_error + 1
        i = 0
        while abs(error_avg) > stop_error:
            print(f"epoch {i}")
            print(f"{self.weights}, {self.bias}")
            i += 1
            total_input = self.sum_input(x_train)
            errors = (y_train - total_input)
            step = x_train.T.dot(errors)
            self.weights += self.alfa * step
            self.bias = self.alfa * errors.sum()
            errors_squered = (errors ** 2)
            error_avg = errors_squered.sum() / set_size
            print(f"error: {error_avg}")
