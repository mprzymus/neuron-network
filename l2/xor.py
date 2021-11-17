import numpy as np

from l2.activation_function import Sigmoid, Relu, Tanh
from l2.extension_data import x_train_unipolar_aug, y_train_unipolar_aug
from l2.network import Network


def xor_test():
    model = Network(input_size=2, learning_step=0.01)
    model.add_layer(4, act_function=Sigmoid)
    model.add_layer(4, act_function=Sigmoid)
    model.compile(2)
    x_test = x_train_unipolar_aug
    y_test = y_train_unipolar_aug

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

    #  print("First prediction")
    #  print(f"{model.predict(x_train_unipolar_aug[0])}, {y_train_unipolar_aug[0]}")
    #  print(f"{model.predict(x_train_unipolar_aug[-1])}, {y_train_unipolar_aug[-1]}")
    print("Learning steps:")
    model.fit(x_train_unipolar_aug[:300], y_train_unipolar_aug[:300], verbose=True, x_valid=y_train_unipolar_aug[300:],
              y_valid=y_train_unipolar_aug[300:], max_epochs=25, batch_size=5)
    print("Prediction after learning")

    false_positive = 0
    false_negative = 0
    score = 0
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
