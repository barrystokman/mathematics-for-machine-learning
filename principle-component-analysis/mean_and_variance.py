import numpy as np


def numpify(dataset):
    return np.array(dataset)


def mean(dataset):
    return sum(dataset)/len(dataset)


def np_mean(dataset):
    np_dataset = numpify(dataset)
    return np_dataset.mean()


def variance(dataset):
    data_mean = mean(dataset)
    variance = sum([((datapoint - data_mean) ** 2 / len(dataset)) for datapoint in dataset])
    return variance


def np_variance(dataset):
    np_dataset = numpify(dataset)
    return np_dataset.var()


def standard_deviation(dataset):
    return variance(dataset) ** (1/2)


def np_standard_deviation(dataset):
    np_dataset = numpify(dataset)
    return np_dataset.std()


def mean_after_adding_datapoint(old_mean, n, new_datapoint):
    return old_mean + (new_datapoint - old_mean) / n


def standard_deviation_after_adding_datapoint(old_sd, old_mean, n, new_datapoint):
    new_mean = mean_after_adding_datapoint(old_mean, n, new_datapoint)
    return (((n - 1) / n) * (old_sd ** 2) + ((new_datapoint - old_mean) * (new_datapoint -
                                                                           new_mean)) / n) ** 0.5


dataset_q1 = [1, 2, 3, 2]
dataset_q3 = [datapoint + 1 for datapoint in dataset_q1]
dataset_q4 = [datapoint * 2 for datapoint in dataset_q1]
dataset_compare = [1, 2, 3, 2, 4]
new_datapoint = 4
n = 5
old_mean = np_mean(dataset_q1)
old_sd = np_standard_deviation(dataset_q1)

# print(f"Mean of dataset_q1: {np_mean(dataset_q1)}")
# print(f"Variance of dataset_q1: {np_variance(dataset_q1)}")
# print(f"Standard deviation of datset_q1: {np_standard_deviation(dataset_q1)}")
# print(f"Variance of dataset_q3: {np_variance(dataset_q3)}")
# print(f"Variance of dataset_q4: {np_variance(dataset_q4)}")
# print(f"Var(dataset_q4) / Var(dataset_q1): {np_variance(dataset_q4) / np_variance(dataset_q1)}")
print(mean_after_adding_datapoint(old_mean, n, new_datapoint), np_mean(dataset_compare))
print(standard_deviation_after_adding_datapoint(old_sd, old_mean, n, new_datapoint), np_standard_deviation(dataset_compare))

