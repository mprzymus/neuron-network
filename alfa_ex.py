import numpy as np

from extension_data import x_train_unipolar_aug, y_train_unipolar_aug
from output_utils import show_history
from perceptron import Perceptron


def weights_perceptron(alfa):
    return Perceptron(alfa=alfa)


def conduct(neural_factory):
    for alfa in np.linspace(10, 100, 3):
        history = []
        for i in range(10):
            perceptron = neural_factory(alfa)
            perceptron.fit(x_train_unipolar_aug, y_train_unipolar_aug, verbose=False)
            history.append(perceptron.epochs)
        show_history(history, round(alfa*1000)/1000)


if __name__ == '__main__':
    conduct(weights_perceptron)
