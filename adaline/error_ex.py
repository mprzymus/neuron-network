import numpy as np

from adaline import Adaline
from extension_data import x_train_bipolar_aug, y_train_bipolar_aug, x_test_bipolar_aug, y_test_bipolar_aug
from adaline.output_utils import show_stats


def weights_adaline(alfa):
    return Adaline(alfa=alfa)


def conduct(neural_factory):
    for error in np.linspace(0.09, 0.5, 20):
        false_positive = 0
        false_negative = 0
        true_positive = 0
        true_negative = 0
        for i in range(10):
            model = neural_factory(error)
            model.fit(x_train_bipolar_aug, y_train_bipolar_aug, stop_error=0.09, verbose=False)
            for x, y in zip(x_test_bipolar_aug, y_test_bipolar_aug):
                y_pred = model.predict(x)
                if y_pred == 1 and y == 1:
                    true_positive += 1
                if y_pred == -1 and y == 1:
                    false_negative += 1
                if y_pred == 1 and y == -1:
                    false_positive += 1
                if y_pred == -1 and y == -1:
                    true_negative += 1

        show_stats([true_positive / 10, true_negative / 10, false_negative / 10, false_positive / 10],
                   round(error * 1000) / 1000)


if __name__ == '__main__':
    conduct(weights_adaline)
