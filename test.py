from unittest import TestCase
import numpy
from preprocess import *


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
        assert encode_one_hot(get_raw_labels()).shape == (6500, 10)


class GroupByLabelTest(TestCase):
    def runTest(self):
        data = numpy.array([[1], [2], [3], [4], [5]])
        labels = numpy.array([0, 1, 0, 1, 2])
        grouped = group_by_label((data, labels))
        assert (grouped[0] == numpy.array([[1], [3]])).all()
        assert (grouped[1] == numpy.array([[2], [4]])).all()
        assert (grouped[2] == numpy.array([[5]])).all()


class RandomPartitionTest(TestCase):
    def runTest(self):
        parts = random_partition(6, (3/6, 2/6, 1/6))
        assert len(parts[0]) == 3
        assert len(parts[1]) == 2
        assert len(parts[2]) == 1


class RandomPartitionTooFewTest(TestCase):
    def runTest(self):
        parts = random_partition(1, [1/2, 1/2])
        assert len(parts[0]) == 1
        assert len(parts) == 1


class SplitByPartsTest(TestCase):
    def runTest(self):
        data = numpy.array([1, 2, 3, 4])
        parts = [[0, 2], [1, 3]]
        split = split_by_parts(data, parts)
        assert (split[0] == numpy.array([1, 3])).all()
        assert (split[1] == numpy.array([2, 4])).all()


class SplitByParts2DTest(TestCase):
    def runTest(self):
        data = numpy.array([[1], [2], [3], [4]])
        parts = [[0, 2], [1, 3]]
        split = split_by_parts(data, parts)
        assert (split[0] == numpy.array([[1], [3]])).all()
        assert (split[1] == numpy.array([[2], [4]])).all()


class StratifyDataTest(TestCase):
    def runTest(self):
        data = numpy.array([[1, 9], [2, 9], [3, 9], [4, 9], [5, 9], [6, 9], [7, 9], [8, 9]])
        labels = numpy.array([0, 0, 0, 1, 1, 1, 2, 2])
        percentages = (1/2, 1/2)
        stratified = stratify_data((data, labels), percentages)
        print(stratified)
        assert stratified[0][0].shape[0] == 5
        assert (stratified[0][1] == numpy.array([0, 0, 1, 1, 2])).all()
        assert stratified[1][0].shape[0] == 3
        assert (stratified[1][1] == numpy.array([0, 1, 2])).all()

