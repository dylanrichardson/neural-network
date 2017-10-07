from run_model import run_model, calc_derivations
import numpy
from tqdm import tqdm
import os


default_params = {
    'epochs': 100,
    'layers': 1,
    'neurons': 10,
    'batch': 512,
    'weight': 'he_normal',
    'activation': 'relu',
    'verbose': 0
}


opt_params = {
    'epochs': 90,
    'layers': 5,
    'neurons': 60,
    'batch': 300,
    'weight': 'glorot_normal',
    'activation': 'relu'
}


default_param_ranges = {
    'epochs': list(range(1,10)) + list(range(10, 101, 10)),
    'layers': list(range(1, 11)),
    'neurons': list(range(1,10)) + list(range(10, 101, 10)),
    'batch': list(range(100, 1000, 100)) + list(range(1000, 6001, 1000)),
    'weight': ['he_normal', 'he_uniform', 'lecun_uniform', 'lecun_normal', 'glorot_normal', 'glorot_uniform'],
    'activation': ['relu', 'selu', 'tanh', 'elu', 'sigmoid'],
}


def optimize(params, ranges, iterations, save_file):
    optimal_params = params
    os.makedirs(save_file)
    for i in tqdm(range(iterations), desc='Optimization iterations'):
        param_value_accuracies = experiment_ranges(optimal_params, ranges, save_file + '/' + str(i) + '.npy')
        for param, (param_values, accuracies) in param_value_accuracies.items():
            optimal_param_value = max_param(param_values, accuracies)
            optimal_params[param] = optimal_param_value
    return optimal_params


def max_param(param_values, accuracies):
    return param_values[accuracies.index(max(accuracies))]


def make_param_ranges(params1, params2):
    param_ranges = {}
    for param in params1:
        param_ranges[param] = [params1[param]]
    for param in params2:
        if param in param_ranges:
            param_ranges[param].append(params2[param])
        else:
            param_ranges[param] = [params2[param]]
    return param_ranges


def experiment_ranges(params, ranges, save_file):
    param_value_accuracies = {}
    for param, param_values in tqdm(ranges.items(), desc='Parameter ranges'):
        print(param)
        accuracies = experiment(params, param, param_values)
        param_value_accuracies[param] = param_values, accuracies
        numpy.save(save_file, {'params': params, 'param_value_accuracies': param_value_accuracies})
    return param_value_accuracies


def experiment(params, param, param_values):
    accuracies = []
    new_params = {**default_params, **params}
    for value in tqdm(param_values, desc='Parameter values'):
        new_params[param] = value
        print(value)
        _, _, truth_values, predictions = run_model(new_params)
        accuracy, _, _ = calc_derivations(truth_values, predictions)
        accuracies.append(accuracy)
    return accuracies


def main():
    optimal_params = optimize(opt_params, default_param_ranges, 2, 'optimize1')
    print(optimal_params)


if __name__ == "__main__":
    main()
