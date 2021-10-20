import numpy as np

from adaline import Adaline
from adaline.extension_data import y_train_bipolar_aug, x_train_bipolar_aug
from adaline.output_utils import show_history


def weights_perceptron(max_weight):
    return Adaline(max_weight=max_weight)


def conduct(neural_factory):
    for weight in np.linspace(-1, 1, 11):
        history = []
        for i in range(10):
            model = neural_factory(weight)
            model.fit(x_train_bipolar_aug, y_train_bipolar_aug, stop_error=0.11)
            history.append(model.epochs)
            errors = 0
            for count, x in enumerate(x_train_bipolar_aug):
                if model.predict(x) != y_train_bipolar_aug[count]:
                    errors += 1
            print(errors)
        show_history(history, weight)


if __name__ == '__main__':
    conduct(weights_perceptron)
