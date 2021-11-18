from unittest import TestCase

import numpy as np
from numpy.testing import assert_array_equal

from l2.activation_function import Relu
from l2.network import Network


class TestNetwork(TestCase):
    def setUp(self) -> None:
        self.model = Network(2)
        self.model.add_layer(3, Relu)
        self.model.add_layer(2, Relu)
        self.model.compile(2)

    def test_should_have_expected_size(self):
        self.assertEqual(np.shape(self.model.softmax.weights), (2, 2))
        self.assertIsNotNone(self.model.softmax.bias)
        self.assertEqual(len(self.model.layers), 2)
        self.assertEqual(np.shape(self.model.layers[0].weights), (3, 2))
        self.assertEqual(np.shape(self.model.layers[1].weights), (2, 3))

    def test_forward_prop_should_return_0_for_0_in_relu_0_bias(self):
        # given
        for layer in self.model.layers:
            layer.bias = 0
        xs = np.array([0, 0])

        # when
        prediction = self.model.predict(xs)

        # then
        assert_array_equal(np.array([0.5, 0.5]), prediction)
        assert_array_equal(np.zeros(shape=2), self.model.softmax.last_input)

    def test_should_gradient_clip(self):
        self.model.gradient_clip = 2

        gradient = np.array([-1000, 1, -1, 0, 10000])
        expected = np.array([-2, 1, -1, 0, 2])

        clipped = self.model.clip_gradient(gradient)

        assert_array_equal(expected, clipped)

    def test_should_forward_prop(self):
        self.set_fixed_weights()

        self.model.predict(np.array([1, 2]))

        assert_array_equal(np.array([1, 2]), self.model.layers[0].last_input)
        assert_array_equal(np.array([0.5, 1.1, 1.7]), self.model.layers[0].last_result)
        assert_array_equal(np.array([0.5, 1.1, 1.7]), self.model.layers[1].last_input)
        assert_array_equal(np.array([12.3, 15.6]), self.model.layers[1].last_result)
        assert_array_equal(np.array([12.3, 15.6]), self.model.softmax.last_input)
        assert_array_equal(np.array([43.5, 40.2]), self.model.softmax.last_result)

    def set_fixed_weights(self):
        self.model.layers[0].weights = np.array([
            [0.1, 0.2],
            [0.3, 0.4],
            [0.5, 0.6],
        ])
        self.model.layers[1].weights = np.array([
            [1, 3, 5],
            [2, 4, 6],
        ])
        self.model.softmax.weights = np.array([
            [1, 2],
            [2, 1],
        ])
        self.model.softmax.bias = 0
        self.model.layers[0].bias = 0
        self.model.layers[1].bias = 0
