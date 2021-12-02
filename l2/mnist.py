import numpy as np

from l2.activation_function import Sigmoid
from l2.alfa_opt import Adam
from l2.network import Network
from l2.weight_init_factory import XavierWeightInitStrategy
from mnist_prep import prepare_mnist_1d

train_size = 50000


def mnist_test():
    x_train, y_train, x_valid, y_valid, x_test, y_test = prepare_mnist_1d(train_size)
    model = Network(input_size=784, learning_step=0.01, gradient_clip=1, optimizer=Adam())
    model.add_layer(13, act_function=Sigmoid, weights_init_strategy=XavierWeightInitStrategy())
    model.add_layer(12, act_function=Sigmoid, weights_init_strategy=XavierWeightInitStrategy())
    model.add_layer(11, act_function=Sigmoid, weights_init_strategy=XavierWeightInitStrategy())
    model.compile(10)
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
              y_valid=y_valid, batch_size=10, max_epochs=6, target_loss=0.15)
    print("Prediction after learning")

    score = 0
    for x, y in zip(x_test, y_test):
        y_predict = model.predict(x)
        if np.argmax(y_predict) == np.argmax(y):
            score += 1
    print(f"score percentage: {score / len(x_test)}")


if __name__ == '__main__':
    mnist_test()
