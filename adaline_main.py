import numpy as np

from activation_function import threshold_bipolar, threshold_unipolar
from adaline import Adaline


def extend_data(x_train, y_train, factor):
    array_x = []
    array_y = []
    for count, x in enumerate(x_train):
        array_x.append(x)
        array_y.append(y_train[count])
        for i in range(factor):
            new_data = x + (np.random.rand(np.size(x)) * 0.01)
            array_x.append(new_data)
            array_y.append(y_train[count])
    return np.array(array_x), np.array(array_y)


x_train_unipolar = np.array([[1, 1], [1.1, 1], [1, 1.1], [1, 0], [0, 1], [0, 0]])
y_train_unipolar = np.array([1, 1, 1, 0, 0, 0])
x_train_bipolar = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]])
y_train_bipolar = np.array([1, -1, -1, -1])

if __name__ == '__main__':
    model = Adaline()
    #  x_train_extended, y_train_extended = extend_data(x_train_bipolar, y_train_bipolar, 5)
    model.fit(x_train_bipolar, y_train_bipolar)
    for x in x_train_bipolar:
        print(model.predict(x, verbose=False))
