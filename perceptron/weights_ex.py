import statistics

import numpy as np

from extension_data import extend_data, x_train_unipolar_aug, y_train_unipolar_aug
from output_utils import show_history
from perceptron import Perceptron


def weights_perceptron(max_weight):
    return Perceptron(bias_static=False, bias=0, max_weight=max_weight)


def conduct(neural_factory):
    for weight in np.linspace(-1, 1, 11):
        history = []
        for i in range(10):
            perceptron = neural_factory(weight)
            perceptron.fit(x_train_unipolar_aug, y_train_unipolar_aug, verbose=False)
            history.append(perceptron.epochs)
        show_history(history, weight)


if __name__ == '__main__':
    conduct(weights_perceptron)
