import numpy as np

from l2.activation_function import Sigmoid, Relu, Tanh
from l2.extension_data import x_train_unipolar_aug, y_train_unipolar_aug, y_train_unipolar_test, x_train_unipolar_test
from l2.network import Network
from l2.score import count_stats


def xor_test():
    model = Network(input_size=2, learning_step=0.01)
    model.add_layer(10, act_function=Relu)
    model.add_layer(4, act_function=Relu)
    model.compile(2)
    x_test = x_train_unipolar_aug
    y_test = y_train_unipolar_aug

    count_stats(model, x_test, y_test)
    count_stats(model, x_train_unipolar_test, y_train_unipolar_test)

    #  print("First prediction")
    #  print(f"{model.predict(x_train_unipolar_aug[0])}, {y_train_unipolar_aug[0]}")
    #  print(f"{model.predict(x_train_unipolar_aug[-1])}, {y_train_unipolar_aug[-1]}")
    print("Learning steps:")
    model.fit(x_test, y_test, verbose=True, x_valid=x_test, target_loss=0.001,
              y_valid=y_test, max_epochs=100, batch_size=1)
    print("Prediction after learning")

    count_stats(model, x_test, y_test)
    count_stats(model, x_train_unipolar_test, y_train_unipolar_test)


if __name__ == '__main__':
    xor_test()
