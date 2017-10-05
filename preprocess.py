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


def encode_one_hot(raw_labels):
    return keras.utils.to_categorical(raw_labels, num_classes=10)


def decode_one_hot(one_hot):
    return int(numpy.argmax(one_hot))


def group_by_label(data_labels):
    grouped = {}
    for (data, label) in zip(data_labels[0], data_labels[1]):
        if label in grouped:
            grouped[label] = numpy.concatenate((grouped[label], numpy.array([data])))
        else:
            grouped[label] = numpy.array([data])
    return grouped


def random_partition(size, percentages):
    parts = [[] for _ in range(min(len(percentages), size))]
    indices = list(range(size))
    for (pIndex, percentage) in enumerate(percentages):
        while len(indices) > 0 and len(parts[pIndex]) < percentage * size:
            index = randint(0, len(indices) - 1)
            parts[pIndex].append(indices[index])
            del indices[index]
    return parts


def split_by_parts(data, parts):
    split = [None for _ in range(len(parts))]
    for (pIndex, part) in enumerate(parts):
        for index in part:
            if split[pIndex] is not None:
                split[pIndex] = numpy.concatenate((split[pIndex], numpy.array([data[index]])))
            else:
                split[pIndex] = numpy.array([data[index]])
    return split


def empty_array(shape_tuple):
    shape_list = list(shape_tuple)
    shape_list[0] = 0
    return numpy.empty(shape=tuple(shape_list), dtype=int)


def array_append(arr1, arr2):
    return [numpy.concatenate((x, y)) for x, y in zip(arr1, arr2)]


def stratify_data(data_labels, percentages):
    grouped = group_by_label(data_labels)
    shape = data_labels[0].shape
    strat_data = [empty_array(shape) for _ in range(len(percentages))]
    strat_labels = [empty_array((shape[0],)) for _ in range(len(percentages))]
    for label in grouped:
        parts = random_partition(grouped[label].shape[0], percentages)
        split_data = split_by_parts(grouped[label], parts)
        split_labels = split_by_parts(numpy.full(grouped[label].shape[0], label), parts)
        strat_data = array_append(strat_data, split_data)
        strat_labels = array_append(strat_labels, split_labels)
    return [(x, y) for x, y in zip(strat_data, strat_labels)]


def preprocess():
    images = get_images()
    raw_labels = get_raw_labels()
    stratified = stratify_data((images, raw_labels), (0.6, 0.25, 0.15))
    for i in range(len(stratified)):
        data_labels = list(stratified[i])
        data_labels[1] = encode_one_hot(data_labels[1])
        stratified[i] = tuple(data_labels)
    return tuple(stratified)
