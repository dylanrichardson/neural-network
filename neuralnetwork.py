from preprocess import preprocess, decode_one_hot
from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy
import argparse


def train_model(model, training_set, validation_set, epochs):
    history = model.fit(training_set[0], training_set[1],
                        validation_data=(validation_set[0], validation_set[1]),
                        epochs=epochs,
                        batch_size=512,
                        verbose=0)
    return history


def make_model(num_layers, num_neurons, activation):
    model = Sequential()
    for _ in range(num_layers):
        model.add(Dense(num_neurons, input_shape=(784,), kernel_initializer='he_normal'))
        model.add(Activation(activation))
    model.add(Dense(num_neurons, kernel_initializer='he_normal'))  # last layer
    model.add(Activation('softmax'))
    model.compile(optimizer='sgd',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def test_model(model, test_set):
    confusion_matrix = numpy.zeros(shape=(10, 10), dtype=int)
    predictions = model.predict(test_set[0])
    for (index, prediction) in enumerate(predictions):
        confusion_matrix[decode_one_hot(prediction)][decode_one_hot(test_set[1][index])] += 1
    # TODO create confusion matrix 10x10
    return confusion_matrix


def save_to_file(file, data):
    numpy.save(file, data)


def parse_args():
    parser = argparse.ArgumentParser(description='Train and test neural network')
    parser.add_argument('--epochs', '-e', type=int, default=100,
                        help='the number of epochs to train with')
    parser.add_argument('--layers', '-l', type=int, default=1,
                        help='the number of layers in the model')
    parser.add_argument('--neurons', '-n', type=int, default=10,
                        help='the number of neurons in each layer of the model')
    parser.add_argument('--activation', '-a', default='relu',
                        help='the activation function for each layer of the model')
    parser.add_argument('--history', default='history.npy',
                        help='the file for saving the training accuracy')
    parser.add_argument('--confusion', '-c', default='confusion.npy',
                        help='the file for saving the confusion matrix')

    return parser.parse_args()


def main():
    args = parse_args()
    (training, validation, testing) = preprocess()
    model = make_model(args.layers, args.neurons, args.activation)

    history = train_model(model, training, validation, args.epochs)
    save_to_file('history.npy', history.history)

    confusion_matrix = test_model(model, testing)
    save_to_file('confusion.npy', confusion_matrix)


if __name__ == "__main__":
    main()
