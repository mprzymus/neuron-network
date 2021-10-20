import numpy as np

from adaline import Adaline
from l1.extension_data import extend_data

x_train_bipolar = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]])
y_train_bipolar = np.array([1, -1, -1, -1])

if __name__ == '__main__':
    model = Adaline()
    x_train_bipolar, y_train_bipolar = extend_data(x_train_bipolar, y_train_bipolar, 5)
    model.fit(x_train_bipolar, y_train_bipolar)
    x_test, y_test = extend_data(x_train_bipolar, y_train_bipolar, 2)
    for count, x in enumerate(x_test):
        if model.predict(x) != y_test[count]:
            print(f"error for: {x}")
    print(f"wynik: {model.predict([1.1, -1.1])}")
