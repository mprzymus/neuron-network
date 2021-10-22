import unittest

import numpy as np
from numpy.testing import assert_array_equal

from l2.layers import Layer

LAYER_SIZE = 6

INPUT_SIZE = 4


class OnesInitStrategy:
    def init_weights(self, input_size, layer_size):
        return np.arange(input_size*layer_size).reshape((layer_size, input_size))


class TestLayer(unittest.TestCase):
    def setUp(self) -> None:
        self.layer = Layer(INPUT_SIZE, LAYER_SIZE, weights_init_strategy=OnesInitStrategy())

    def test_activate(self):
        input_vector = np.array([1, 2, 3, 4])

        result = self.layer.activate(input_vector)

        assert result.shape == (LAYER_SIZE,)
        assert_array_equal(result, np.array([20, 60, 100, 140, 180, 220]))
