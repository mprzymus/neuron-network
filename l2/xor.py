import numpy as np

from l2.activation_function import Sigmoid, Relu, Tanh
from l2.extension_data import x_train_unipolar_aug, y_train_unipolar_aug, y_train_unipolar_test, x_train_unipolar_test
from l2.network import Network


def xor_test():
    model = Network(input_size=2, learning_step=0.01)
    model.add_layer(10, act_function=Relu)
    model.add_layer(4, act_function=Relu)
    model.compile(2)
    x_test = x_train_unipolar_aug
    y_test = y_train_unipolar_aug

    score_print(model, x_test, y_test)
    score_print(model, x_train_unipolar_test, y_train_unipolar_test)

    #  print("First prediction")
    #  print(f"{model.predict(x_train_unipolar_aug[0])}, {y_train_unipolar_aug[0]}")
    #  print(f"{model.predict(x_train_unipolar_aug[-1])}, {y_train_unipolar_aug[-1]}")
    print("Learning steps:")
    model.fit(x_test, y_test, verbose=True, x_valid=x_test, target_loss=0.001,
              y_valid=y_test, max_epochs=100, batch_size=1)
    print("Prediction after learning")

    score_print(model, x_test, y_test)
    score_print(model, x_train_unipolar_test, y_train_unipolar_test)


def score_print(model, x_test, y_test):
    score = 0
    false_positive = 0
    false_negative = 0
    for x, y in zip(x_test, y_test):
        y_predict = model.predict(x)
        if np.argmax(y_predict) == np.argmax(y):
            score += 1
        elif np.argmax(y_predict) == 0:
            false_positive += 1
        else:
            false_negative += 1
    print(f"score percentage: {score / len(x_test)}")
    print(false_negative)
    print(false_positive)


if __name__ == '__main__':
    xor_test()
