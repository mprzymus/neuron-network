import numpy as np


def extend_data(x_train, y_train, augmented_size_factor, element_diff_factor=0.01):
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
