import numpy as np
from tensorflow import keras

from l2.activation_function import Sigmoid, Relu, Tanh
from l2.momentum import Momentum, Nag, MomentumStrategy
from l2.network import Network

train_size = 5000


def mnist_test():
    x_train, y_train, x_valid, y_valid, x_test, y_test = prepare_data()
    model = Network(input_size=784, learning_step=0.05, gradient_clip=10)
    model.add_layer(10, act_function=Relu)
    model.add_layer(5, act_function=Relu)
    model.compile(10, momentum=Nag(), momentum_rate=0.5)
    score = 0
    for x, y in zip(x_test, y_test):
        y_predict = model.predict(x)
        if np.argmax(y_predict) == np.argmax(y):
            score += 1
    print(f"score percentage: {score / len(x_test)}")

    #  print("First prediction")
    #  print(f"{model.predict(x_train_unipolar_aug[0])}, {y_train_unipolar_aug[0]}")
    #  print(f"{model.predict(x_train_unipolar_aug[-1])}, {y_train_unipolar_aug[-1]}")
    print("Learning steps:")
    model.fit(x_train, y_train, verbose=True, x_valid=x_valid,
              y_valid=y_valid, batch_size=100, max_epochs=6, target_loss=0.15)
    print("Prediction after learning")

    score = 0
    for x, y in zip(x_test, y_test):
        y_predict = model.predict(x)
        if np.argmax(y_predict) == np.argmax(y):
            score += 1
    print(f"score percentage: {score / len(x_test)}")


def prepare_data():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
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


if __name__ == '__main__':
    mnist_test()
