from abc import ABC, abstractmethod
from math import sqrt

import numpy as np


class WeightInitStrategy(ABC):
    @abstractmethod
    def init_weights(self, input_size, output_size):
        pass


class XavierWeightInitStrategy(WeightInitStrategy):
    def __init__(self, mean=0.0):
        self.gaussian_initializer = GaussianWeightsInitStrategy(mean)

    def init_weights(self, input_size, output_size):
        variance = 2 / input_size + output_size
        return np.random.normal(loc=0, scale=sqrt(variance), size=(output_size, input_size))


class HeWeightInitStrategy(WeightInitStrategy):
    def init_weights(self, input_size, output_size):
        variance = 2 / input_size
        return np.random.normal(loc=0, scale=sqrt(variance), size=(output_size, input_size))


class GaussianWeightsInitStrategy(WeightInitStrategy):
    def __init__(self, mean=0.0, standard_dev=0.1):
        self.loc = mean
        self.scale = standard_dev

    def init_weights(self, input_size, layer_size):
        return np.random.normal(size=(layer_size, input_size), loc=self.loc, scale=self.scale)

    def __str__(self):
        return f"Gaussian, mean={self.loc}, scale={self.scale}"


class FixedWeightInitStrategy(WeightInitStrategy):
    def __init__(self, value=0):
        self.value = value

    def init_weights(self, input_size, layer_size):
        return np.zeros(shape=(layer_size, input_size)) + self.value

    def __str__(self):
        return f"Fixed, value={self.value}"
