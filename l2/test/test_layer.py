from unittest import TestCase

import numpy as np
from numpy.testing import *

from l2.layers import Layer, Softmax

BIAS = 5

LAYER_SIZE = 6

INPUT_SIZE = 4


class SequentialInitStrategy:
    @staticmethod
    def init_weights(input_size, layer_size):
        return np.arange(input_size * layer_size).reshape((layer_size, input_size))


class TestLayer(TestCase):
    def setUp(self) -> None:
        self.layer = Layer(INPUT_SIZE, LAYER_SIZE, weights_init_strategy=SequentialInitStrategy(), bias=BIAS)

    def test_activate(self):
        input_vector = np.array([1, 2, 3, 4])

        result = self.layer.activate(input_vector)

        assert result.shape == (LAYER_SIZE,)
        assert_array_equal(result, np.array([20 + BIAS, 60 + BIAS, 100 + BIAS, 140 + BIAS, 180 + BIAS, 220 + BIAS]))
        assert_array_equal(self.layer.last_result,
                           np.array([20 + BIAS, 60 + BIAS, 100 + BIAS, 140 + BIAS, 180 + BIAS, 220 + BIAS]))


class TestSoftmaxFunction(TestCase):
    def setUp(self) -> None:
        self.softmax = Softmax.SoftmaxFun()

    def test_1d(self):
        input_array = np.array([1, 2, 3, 6])

        result = self.softmax.apply(input_array)

        assert_array_almost_equal(np.array([0.006269, 0.01704, 0.04632, 0.93037]), result)

    def test_2d(self):
        input_array = np.array([[1, 2, 3, 6], [1, 3, 3, 4]])

        result = self.softmax.apply(input_array)

        print(result)

        assert_array_almost_equal(
            np.array([[0.006269, 0.01704, 0.04632, 0.93037], [0.02788339, 0.20603191, 0.20603191, 0.56005279]]), result)
