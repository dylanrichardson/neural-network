import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from PIL import Image


def make_training_graph(training, validation):
    fig = plt.figure()
    plt.plot(training, label='training')
    plt.plot(validation, label='validation')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()
    return fig


def make_testing_table(truth_values, predictions):
    precision = precision_score(truth_values, predictions, average=None)
    recall = recall_score(truth_values, predictions, average=None)
    nrows, ncols = 2+1, 10
    hcell, wcell = 0.3, 1.
    hpad, wpad = 0, 0
    fig=plt.figure(figsize=(ncols*wcell+wpad, nrows*hcell+hpad))
    ax = fig.add_subplot(111)
    ax.axis('off')
    plt.table(cellText=np.stack((precision, recall)),
              rowLabels=['precision', 'recall'],
              colLabels=range(1, 11),
              loc='center',
              fontsize=20)
    return fig


def make_confusion_matrix(truth_values, predictions):
    data = confusion_matrix(truth_values, predictions)

    fig = plt.figure()
    plt.imshow(data, cmap='Oranges')
    ax = plt.gca()
    for (x, y), value in np.ndenumerate(data):
        ax.text(x, y, value, va='center', ha='center')
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    digits = np.arange(0, 10, 1)
    ax.set_xticks(digits)
    ax.set_yticks(digits)
    ax.set_xticklabels(digits)
    ax.set_yticklabels(digits)
    ax.set_xlabel('predicted')
    ax.set_ylabel('actual')
    return fig


def make_image(image_array):
    w, h = image_array.shape
    rgb = np.zeros((w * h, 3), dtype=np.uint8)
    img = image_array.reshape((w * h,))
    for index, value in enumerate(img):
        rgb[index] = [value] * 3
    rgb = rgb.reshape((w, h, 3))
    return Image.fromarray(rgb)


def make_bar_graph(values, accuracies):
    fig = plt.figure()
    y_pos = np.arange(len(values))
    plt.bar(y_pos, accuracies)
    plt.xticks(y_pos, values, rotation=30)
    return fig


def make_line_graph(values, accuracies):
    fig = plt.figure()
    plt.plot(values, accuracies)
    return fig


def make_param_graph(param, values, accuracies):
    if type(values[0]) is str:
        fig = make_bar_graph(values, accuracies)
    else:
        fig = make_line_graph(values, accuracies)
    plt.ylabel('accuracy')
    plt.xlabel(param)
    plt.title('accuracy vs ' + param)
    return fig


def main():
    data = np.load('misclass.npy')
    make_image(data[70]).save('misclassified2.png')
    plt.show()


if __name__ == "__main__":
    main()

