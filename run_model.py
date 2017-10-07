import argparse
import numpy
import warnings
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score

from graph import make_training_graph, make_testing_table, make_confusion_matrix


def save_to_file(file, *args, **kwds):
    numpy.savez(file, *args, **kwds)


def calc_derivations(truth_values, predictions):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        accuracy = accuracy_score(truth_values, predictions)
        precision = precision_score(truth_values, predictions, average=None)
        recall = recall_score(truth_values, predictions, average=None)
    return accuracy, precision, recall


def save_misclassified(file, truth_values, predictions, test_set):
    images = []
    for index, (prediction, truth_value) in enumerate(zip(predictions, truth_values)):
        if prediction != truth_value:
            images.append(test_set[index])
    numpy.save(file, images)


def run_model(params, callbacks=None):
    if callbacks is None:
        callbacks = []
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
    import sys
    stderr = sys.stderr
    sys.stderr = open('/dev/null', 'w')
    from neural_network import make_model, train_model, test_model
    from preprocess import preprocess, decode_one_hot_array
    sys.stderr = stderr

    training, validation, testing = preprocess()
    test_set = testing[0]
    truth_values = decode_one_hot_array(testing[1])

    model = make_model(params['layers'], params['neurons'], params['activation'], params['weight'])

    history = train_model(model, training, validation, params['epochs'], params['batch'], params['verbose'], callbacks)

    predictions = test_model(model, test_set)

    if params['misclassified']:
        save_misclassified(params['misclassified'], truth_values, predictions, test_set)

    return model, history.history, truth_values, predictions


def parse_args():
    parser = argparse.ArgumentParser(description='Train and test neural network')
    parser.add_argument('--epochs', '-e', type=int, default=100,
                        help='the number of epochs to train the model with')
    parser.add_argument('--batch', '-b', type=int, default=512,
                        help='the batch size to train the model with')
    parser.add_argument('--layers', '-l', type=int, default=1,
                        help='the number of layers in the model')
    parser.add_argument('--neurons', '-n', type=int, default=10,
                        help='the number of neurons in each layer of the model')
    parser.add_argument('--activation', '-f', default='relu',
                        help='the activation function for each layer of the model')
    parser.add_argument('--weight', '-w', default='he_normal',
                        help='the weight initialization for each layer of the model')
    parser.add_argument('--training',
                        help='the file for saving the training and validation accuracy data')
    parser.add_argument('--testing',
                        help='the file for saving the testing predictions and actual data')
    parser.add_argument('--model', '-m',
                        help='the file for saving the model')
    parser.add_argument('--misclassified',
                        help='the file for saving the misclassified images')
    parser.add_argument('--verbose', '-v', action='count',
                        help='print out information')
    parser.add_argument('--params',
                        help='the file for loading parameters')
    parser.add_argument('--graph', '-g',
                        help='the file for saving graphs')
    return parser.parse_args()


def main():
    args = parse_args()
    params = vars(args)

    if args.params:
        params = {**params, **numpy.load(args.params).item()}

    model, history, truth_values, predictions = run_model(params)

    if args.model:
        model.save(args.model)

    if args.training:
        save_to_file(args.training, training=history['acc'], validation=history['val_acc'])

    if args.testing:
        save_to_file(args.testing, truth_values=truth_values, predictions=predictions)

    if args.graph:
        os.makedirs(args.graph)
        make_training_graph(history).save_fig(args.graph + '/training.png')
        make_testing_table(truth_values, predictions).save_fig(args.graph + '/testing.png')
        make_confusion_matrix(truth_values, predictions).save_fig(args.graph + '/confusion_matrix.png')

    accuracy, precision, recall = calc_derivations(truth_values, predictions)
    print('accuracy:', accuracy)
    print('precision:', precision)
    print('recall:', recall)


if __name__ == "__main__":
    main()