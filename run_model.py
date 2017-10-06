import argparse
import numpy
import warnings
from sklearn.metrics import accuracy_score, precision_score, recall_score


def save_to_file(file, *args, **kwds):
    numpy.savez(file, *args, **kwds)


def calc_derivations(true_values, predictions):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        accuracy = accuracy_score(true_values, predictions)
        precision = precision_score(true_values, predictions, average=None)
        recall = recall_score(true_values, predictions, average=None)
    return accuracy, precision, recall


def run_model(params):
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

    history = train_model(model, training, validation, params['epochs'], params['batch'], params['verbose'])

    predictions = test_model(model, test_set)

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
    parser.add_argument('--training', '-t',
                        help='the file for saving the training and validation accuracy data')
    parser.add_argument('--predictions', '-p',
                        help='the file for saving the predictions on the test set')
    parser.add_argument('--model', '-m',
                        help='the file for saving the model')
    parser.add_argument('--verbose', '-v', action='count',
                        help='print out information')
    return parser.parse_args()


def main():
    args = parse_args()

    model, history, truth_values, predictions = run_model(vars(args))

    if args.model is not None:
        model.save(args.model)

    if args.training is not None:
        save_to_file(args.training, training=history['acc'], validation=history['val_acc'])

    accuracy, precision, recall = calc_derivations(truth_values, predictions)

    if args.predictions is not None:
        save_to_file(args.predictions, accuracy=accuracy, precision=precision, recall=recall)

    if args.verbose:
        print('accuracy:', accuracy)
        print('precision:', precision)
        print('recall:', recall)


if __name__ == "__main__":
    main()