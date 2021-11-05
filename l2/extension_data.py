import numpy as np


def extend_data(x_train, y_train, augmented_size_factor, element_diff_factor=0.1):
    array_x = []
    array_y = []
    for count, x in enumerate(x_train):
        array_x.append(x)
        array_y.append(y_train[count])
        for i in range(augmented_size_factor):
            new_data = x + (np.random.rand(np.size(x)) * element_diff_factor)
            array_x.append(new_data)
            array_y.append(y_train[count])
    return np.array(array_x), np.array(array_y)


x_train_unipolar = np.array([[1, 1], [1.1, 1], [1, 1.1], [1, 0], [0, 1], [0, 0]])
y_train_unipolar = np.array([[1, 0], [1, 0], [1, 0], [0, 1], [0, 1], [0, 1]])
x_train_unipolar_aug, y_train_unipolar_aug = extend_data(x_train_unipolar, y_train_unipolar, 100)

x_train_bipolar = np.array([[1, 1], [1.1, 1], [1, 1.1], [1, -1], [-1, 1], [-1, -1]])
y_train_bipolar = np.array([1, 1, 1, -1, -1, -1])
x_train_bipolar_aug, y_train_bipolar_aug = extend_data(x_train_bipolar, y_train_bipolar, 10)

x_test_bipolar = np.array([[1, 1], [1.01, 1], [1, 1.01], [1, -1], [-1, 1], [-1, -1]])
y_test_bipolar = np.array([1, 1, 1, -1, -1, -1])
x_test_bipolar_aug, y_test_bipolar_aug = extend_data(x_train_bipolar, y_train_bipolar, 3)
