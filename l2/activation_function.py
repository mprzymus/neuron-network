import math


def relu(x):
    return 0 if x < 0 else x


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def tanh(x):
    return 2 / (1 + math.exp(-2 * x)) + 1
