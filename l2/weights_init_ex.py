from statistics import mean, stdev

import numpy as np

from l2.activation_function import Relu
from l2.weight_init_factory import GaussianWeightsInitStrategy, FixedWeightInitStrategy
from mnist_prep import prepare_mnist_1d
from l2.network import Network
from l2.score import count_stats

MAX_EPOCHS = 15

_gaussian1 = GaussianWeightsInitStrategy(mean=0.0, standard_dev=0.1)
_gaussian2 = GaussianWeightsInitStrategy(mean=0.0, standard_dev=0.5)
_gaussian3 = GaussianWeightsInitStrategy(mean=0.0, standard_dev=10)
_fixed0 = FixedWeightInitStrategy(0)
_fixed10 = FixedWeightInitStrategy(10)
_fixed10_negative = FixedWeightInitStrategy(-10)
weight_init_strategies = [_fixed0]
if __name__ == '__main__':
    x_train, y_train, x_valid, y_valid, x_test, y_test = prepare_mnist_1d()
    for weight_init in weight_init_strategies:
        print(f"Weight init str: {weight_init}")
        matrix = np.zeros(shape=(10, 10))
        score = []
        for i in range(1):
            print(f"Attempt {i}")
            model = Network(input_size=784, learning_step=0.005, gradient_clip=1)
            model.add_layer(12, act_function=Relu, weights_init_strategy=weight_init)
            model.compile(10, weights_init_strategy=weight_init)
            verbose = i == 0
            train_errors, valid_errors, epochs = model.fit(x_train, y_train, verbose=verbose, x_valid=x_valid,
                                                           y_valid=y_valid, batch_size=100,
                                                           max_epochs=MAX_EPOCHS,
                                                           target_loss=0.15)
            ex_score, ex_matrix = count_stats(model, x_test, y_test)
            if epochs != MAX_EPOCHS:
                print(f"Early stop: {epochs}")
            score.append(ex_score)
            matrix += ex_matrix
        matrix /= 10
        print(f"{mean(score)}, {stdev(score)}")
        print(matrix.astype('int64'))
