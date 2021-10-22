import numpy as np

from l2.layers import Layer, Softmax

if __name__ == '__main__':
    l1 = Layer(4, 6)
    l2 = Layer(6, 2)
    l3 = Softmax()

    x = np.arange(4)

    l1x = l1.activate(x)
    l2x = l2.activate(l1x)
    y = l3.activate(l2x)
    print(y)
