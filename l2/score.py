import numpy as np


def count_stats(model, x_test, y_test):
    score = 0
    matrix = np.zeros(shape=(10, 10), dtype='int64')
    for x, y in zip(x_test, y_test):
        y_predict = model.predict(x)
        real_class = np.argmax(y)
        predicted_class = np.argmax(y_predict)
        if real_class == predicted_class:
            score += 1
        matrix[real_class][predicted_class] += 1

    return score / len(x_test), matrix.astype('int64')
