from preprocess import decode_one_hot_array
from keras.models import Sequential
from keras.layers import Dense, Activation


def train_model(model, training_set, validation_set, epochs, batch_size, verbosity, callbacks):
    history = model.fit(training_set[0], training_set[1],
                        validation_data=(validation_set[0], validation_set[1]),
                        epochs=epochs,
                        batch_size=batch_size,
                        verbose=verbosity,
                        callbacks=callbacks)
    return history


def make_model(num_layers, num_neurons, activation, weight_init):
    model = Sequential()
    model.add(Dense(num_neurons, input_shape=(784,), kernel_initializer=weight_init)) # first layer
    model.add(Activation(activation))
    for _ in range(num_layers - 1):
        model.add(Dense(num_neurons, kernel_initializer=weight_init))
        model.add(Activation(activation))
    model.add(Dense(10, kernel_initializer=weight_init))  # last layer
    model.add(Activation('softmax'))
    model.compile(optimizer='sgd',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def test_model(model, test_set):
    predictions = model.predict(test_set)
    return decode_one_hot_array(predictions)
