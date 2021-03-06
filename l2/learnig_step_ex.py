from statistics import mean, stdev

import numpy as np

from l2.activation_function import Relu
from mnist_prep import prepare_mnist_1d
from l2.network import Network
from l2.score import count_stats

MAX_EPOCHS = 30

if __name__ == '__main__':
    x_train, y_train, x_valid, y_valid, x_test, y_test = prepare_mnist_1d()
    for learning_rate in [0.0001, 0.001, 0.01, 0.1, 1]:
        print(f"Alfa: {learning_rate}")
        matrix = np.zeros(shape=(10, 10))
        score = []
        for i in range(10):
            print(f"Attempt {i}")
            model = Network(input_size=784, learning_step=learning_rate, gradient_clip=1)
            model.add_layer(12, act_function=Relu)
            model.compile(10)
            verbose = i == 0
            train_errors, valid_errors, epochs = model.fit(x_train, y_train, verbose=verbose, x_valid=x_valid,
                                                           y_valid=y_valid, batch_size=100, max_epochs=MAX_EPOCHS,
                                                           target_loss=0.15)
            ex_score, ex_matrix = count_stats(model, x_test, y_test)
            if epochs != MAX_EPOCHS:
                print(f"Early stop: {epochs}")
            score.append(ex_score)
            matrix += ex_matrix
        matrix /= 10
        print(f"{mean(score)}, {stdev(score)}")
        print(matrix.astype('int64'))
