from activation_function import threshold_bipolar, threshold_unipolar
from extension_data import x_train_unipolar_aug, y_train_unipolar_aug, y_train_unipolar, x_train_unipolar, \
    x_train_bipolar_aug, y_train_bipolar_aug
from output_utils import show_history
from perceptron import Perceptron


def weights_perceptron(act_function):
    return Perceptron(act_function=act_function)


def conduct(neural_factory):
    for act_fun, x, y in [(threshold_unipolar, x_train_unipolar_aug, y_train_unipolar_aug),
                          (threshold_bipolar, x_train_bipolar_aug, y_train_bipolar_aug)]:
        history = []
        for i in range(10):
            perceptron = neural_factory(act_fun)
            perceptron.fit(x, y, verbose=False)
            history.append(perceptron.epochs)
        show_history(history, act_fun.__name__)


if __name__ == '__main__':
    conduct(weights_perceptron)
