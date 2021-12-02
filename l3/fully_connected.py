from tensorflow.keras.optimizers import Adadelta
from tensorflow.keras.losses import categorical_crossentropy
from keras import Sequential
from keras.layers import Dense

from mnist_prep import prepare_mnist_1d

train_size = 50000


def use_network():
    x_train, y_train, x_valid, y_valid, x_test, y_test = prepare_mnist_1d(train_size)
    model = Sequential()
    model.add(Dense(784, activation='relu'))
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
    use_network()
