# TODO flatten input (reshape)
# TODO encode integer as one-hot vector (to_categorical)
# TODO split data into training, validation, and testing (stratified sampling)
# TODO create model

# TODO create confusion matrix 10x10
import numpy
import keras
from random import randint


def flatten(images):
    return numpy.asarray([image.flatten() for image in images])


def get_images():
    raw_images = numpy.load('images.npy')
    return flatten(raw_images)


def get_raw_labels():
    return numpy.load('labels.npy')


def get_labels(raw_labels):
    return keras.utils.to_categorical(raw_labels, num_classes=10)


def split_by_label(data, labels):
    split = {}
    for (datum, label) in zip(data, labels):
        if label in split:
            split[label] = numpy.concatenate((split[label], numpy.array([datum])))
        else:
            split[label] = numpy.array([datum])
    return split


def split(data, percentages):
    split_data = [numpy.array([])] * len(percentages)
    data_left = data
    for (pIndex, percentage) in enumerate(percentages):
        while split_data[pIndex].size < percentage * data.size and data_left.size > 0:
            dIndex = randint(0, data_left.size)
            split_data[pIndex] = numpy.concatenate((split_data[pIndex], numpy.array(data_left[dIndex])))
    return split_data


def split_data(data, raw_labels):
    data_by_label = split_by_label(data, raw_labels)
    for label in data_by_label:
        split_data = split(data_by_label[0], [0.6, 0.25, 0.15])


def preprocess():
    images = get_images()
    raw_labels = get_raw_labels()
    return split_data(images, raw_labels)


def main():
    data = preprocess()


if __name__ == "__main__":
    main()
