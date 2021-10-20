import numpy as np

from adaline import Adaline
from l1.extension_data import x_train_bipolar_aug, y_train_bipolar_aug
from l1.adaline.adaline import show_history


def weights_adaline(alfa):
    return Adaline(alfa=alfa)


def conduct(neural_factory):
    for alfa in np.linspace(0.0945, 0.6, 20):
        history = []
        for i in range(10):
            model = neural_factory(alfa)
            model.fit(x_train_bipolar_aug, y_train_bipolar_aug, stop_error=0.001, verbose=False)
            history.append(model.epochs)


        show_history(history, round(alfa*1000)/1000)


if __name__ == '__main__':
    conduct(weights_adaline)
