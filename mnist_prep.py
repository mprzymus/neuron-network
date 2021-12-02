import numpy as np
from keras.datasets import mnist
from tensorflow import keras


def prepare_mnist_1d(train_size):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = np.reshape(x_train, newshape=(len(x_train), 784))
    x_test = np.reshape(x_test, newshape=(len(x_test), 784))
    y_todo = np.zeros(shape=(len(y_test), 10))
    x_train = x_train.astype('float64')
    x_test = x_test.astype('float64')
    for y, y_keras in zip(y_todo, y_test):
        y[y_keras] = 1
    y_test = y_todo
    y_todo = np.zeros(shape=(len(y_train), 10))
    for y, y_keras in zip(y_todo, y_train):
        y[y_keras] = 1
    y_train = y_todo
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    return x_train[:train_size], y_train[:train_size], x_train[train_size:], y_train[train_size:], x_test, y_test


def prepare_mnist_2d(train_size):
    num_classes = 10

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype('float64')
    x_test = x_test.astype('float64')
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    return x_train[:train_size], y_train[:train_size], x_train[train_size:], y_train[train_size:], x_test, y_test
