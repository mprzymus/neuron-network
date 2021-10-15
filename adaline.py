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
        return (x * self.weights).sum() + self.bias

    def fit(self, x_train, y_train, stop_error=0.15):
        set_size = np.size(x_train)
        error_avg = stop_error + 1
        i = 0
        while abs(error_avg) > stop_error:
            print(f"epoch {i}")
            print(f"{self.weights}, {self.bias}")
            i += 1
            error_avg = 0
            for count, x in enumerate(x_train):
                net_input = self.sum_input(x)
                errors = y_train[count] - net_input
                self.weights += self.alfa * errors * x
                self.bias += self.alfa * errors
                errors_squared = errors ** 2
                error_avg += errors_squared / set_size
            print(f"error: {error_avg}")
