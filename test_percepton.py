from perceptron import *


def test_should_predict_1():
    #  given
    weights = np.array([1, -1])
    perceptron = Perceptron(weights, threshold_unipolar)
    perceptron.bias = 0.4

    #  when
    result = perceptron.predict(np.array([1, 0]))

    #  then
    assert result == 1


def test_should_predict_0():
    #  given
    weights = np.array([-1, 1])
    perceptron = Perceptron(weights, threshold_unipolar)
    perceptron.bias = 0

    #  when
    result = perceptron.predict(np.array([0, 0]))

    #  then
    assert result == 0


def test_should_predict_1_bi():
    #  given
    weights = np.array([1, -1])
    perceptron = Perceptron(weights, threshold_bipolar)
    perceptron.bias = 0.5

    #  when
    result = perceptron.predict(np.array([1, -1]))

    #  then
    assert result == 1


def test_should_predict_minus1_bi():
    #  given
    weights = np.array([-1, 1])
    perceptron = Perceptron(weights, threshold_bipolar)
    perceptron.bias = 0.5

    #  when
    result = perceptron.predict(np.array([1, -1]))

    #  then
    assert result == -1
