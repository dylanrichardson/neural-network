from unittest import TestCase
from neuralnetwork import *
import numpy


# preprocess

class PreprocessTest(TestCase):
    def runTest(self):
        pass


class FlattenTest(TestCase):
    def runTest(self):
        arr = numpy.array([[[1,2], [3,4]], [[5,6], [7,8]]])
        assert arr.shape == (2, 2, 2)
        arr2 = flatten(arr)
        assert arr2.shape == (2, 4)


class GetImagesTest(TestCase):
    def runTest(self):
        assert get_images().shape == (6500, 784)


class GetLabelsTest(TestCase):
    def runTest(self):
        assert get_labels(get_raw_labels()).shape == (6500, 10)


class SplitByLabelTest(TestCase):
    def runTest(self):
        data = numpy.array([[1], [2], [3], [4], [5]])
        labels = numpy.array([0, 1, 0, 1, 2])
        split = split_by_label(data, labels)
        assert (split[0] == numpy.array([[1], [3]])).all()
        assert (split[1] == numpy.array([[2], [4]])).all()
        assert (split[2] == numpy.array([[5]])).all()


class SplitTest(TestCase):
    def runTest(self):
        data = numpy.array([0, 1, 2, 3, 4, 5])
        split_data = split(data, [3/6, 2/6, 1/6])
        assert split_data[0].size == 3
        assert split_data[1].size == 2
        assert split_data[2].size == 1


class SplitDataTest(TestCase):
    def runTest(self):
        pass

