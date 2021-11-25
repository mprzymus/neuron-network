import numpy as np


def extend_data(x_train, y_train, augmented_size_factor, element_diff_factor=0.2):
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


x_train_unipolar = np.array([[1, 1], [1, 0], [0, 1], [0, 0], [1, 1.1], [1.1, 0], [0, 1.1], [0.1, 0]])
y_train_unipolar = np.array([[1, 0], [0, 1], [0, 1], [1, 0], [1, 0], [0, 1], [0, 1], [1, 0]])

x_train_unipolar_test = np.array([[1, 1.1], [1, 0.1], [0, 1.11], [0, -0.1]])
y_train_unipolar_test = np.array([[1, 0], [0, 1], [0, 1], [1, 0]])

x_train_unipolar_aug, y_train_unipolar_aug = extend_data(x_train_unipolar, y_train_unipolar, 100)

indices = np.arange(x_train_unipolar_aug.shape[0])
np.random.shuffle(indices)

x_train_unipolar_aug = x_train_unipolar_aug[indices]
y_train_unipolar_aug = y_train_unipolar_aug[indices]
