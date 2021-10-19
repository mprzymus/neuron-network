import numpy as np

from extension_data import x_train_unipolar_aug, y_train_unipolar_aug
from output_utils import show_history
from perceptron import Perceptron


def weights_perceptron(alfa):
    return Perceptron(bias_static=False, alfa=alfa)


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
