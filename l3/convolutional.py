from keras import Sequential
from keras.layers import Dense, Conv2D, Flatten
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adadelta

from mnist_prep import prepare_mnist_2d

train_size = 50000


def create_network():
    input_shape = (28, 28, 1)
    x_train, y_train, x_valid, y_valid, x_test, y_test = prepare_mnist_2d(train_size)
    model = Sequential()
    model.add(Conv2D(32, activation='relu', input_shape=input_shape, kernel_size=(3, 3)))
    model.add(Flatten())
    model.add(Dense(10, activation='softmax'))

    model.compile(loss=categorical_crossentropy,
                  optimizer=Adadelta(),
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              batch_size=100,
              epochs=5,
              verbose=1,
              validation_data=(x_valid, y_valid))
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"test loss:     {loss}")
    print(f"test accuracy: {accuracy}")


if __name__ == '__main__':
    create_network()
