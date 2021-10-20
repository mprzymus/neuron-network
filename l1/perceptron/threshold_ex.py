import numpy as np

from l1.extension_data import extend_data
from perceptron import Perceptron


def threshold_perceptron(threshold):
    return Perceptron(weights=np.array([0.001, 0.001]), bias_static=True, bias=threshold)


def threshold_perceptron_random_weights(threshold):
    return Perceptron(weights=np.array([0.001, 0.001]), bias_static=True, bias=threshold, max_weight=0.5)


x_train_unipolar = np.array([[1, 1], [1.1, 1], [1, 1.1], [1, 0], [0, 1], [0, 0]])
y_train_unipolar = np.array([1, 1, 1, 0, 0, 0])
x_train_unipolar_aug, y_train_unipolar_aug = extend_data(x_train_unipolar, y_train_unipolar, 10)


def conduct(neural_factory):
    for threshold in range(-10, 0):
        history = []
        for i in range(10):
            perceptron = neural_factory(threshold)
            perceptron.fit(x_train_unipolar_aug, y_train_unipolar_aug, verbose=False)
            history.append(perceptron.epochs)
        print(f"{threshold} & {history} \\\\")


def conduct_random_data(neural_factory):
    for threshold in range(-10, 0):
        history = []
        for i in range(10):
            perceptron = neural_factory(threshold)
            x, y = extend_data(x_train_unipolar, y_train_unipolar, 10)
            perceptron.fit(x, y, verbose=False)
            history.append(perceptron.epochs)
        print(f"{threshold} & {history} \\\\")


if __name__ == '__main__':
    conduct(threshold_perceptron)
    conduct(threshold_perceptron_random_weights)
