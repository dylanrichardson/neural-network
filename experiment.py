from run_model import run_model, calc_derivations
import numpy

default_params = {
    'epochs': 100,
    'layers': 1,
    'neurons': 10,
    'batch': 512,
    'weight': 'he_normal',
    'activation': 'relu',
    'verbose': 0
}


param_ranges = {
    'epochs': range(1, 100000, 10000),
    'layers': range(1, 10),
    'neurons': range(1, 1000, 100),
    'batch': range(1, 100000, 10000),
    'weight': ['he_normal', 'he_uniform', 'lecun_uniform', 'lecun_normal', 'glorot_normal', 'glorot_uniform'],
    'activation': ['relu', 'selu', 'tanh', 'elu', 'sigmoid'],
}


def optimize(params, ranges, iterations, save_file):
    optimal_params = params
    for _ in range(iterations):
        param_value_accuracies = experiment_ranges(optimal_params, ranges, save_file)
        for param, (param_values, accuracies) in param_value_accuracies.items():
            optimal_param_value = max_param(param_values, accuracies)
            optimal_params[param] = optimal_param_value
    return optimal_params


def max_param(param_values, accuracies):
    return param_values[accuracies.index(max(accuracies))]


def experiment_ranges(params, ranges, save_file):
    param_value_accuracies = {}
    for param, param_values in ranges.items():
        accuracies = experiment(params, param, param_values)
        param_value_accuracies[param] = param_values, accuracies
        numpy.save(save_file, param_value_accuracies)
    return param_value_accuracies


def experiment(params, param, param_values):
    accuracies = []
    new_params = {**default_params, **params}
    for value in param_values:
        new_params[param] = value
        _, _, truth_values, predictions = run_model(new_params)
        accuracy, _, _ = calc_derivations(truth_values, predictions)
        accuracies.append(accuracy)
    return accuracies


def main():
    optimal_params = optimize(default_params, param_ranges, 100, 'optimize.npy')
    print(optimal_params)


if __name__ == "__main__":
    main()
