import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def sd_filter(to_filter, max_sds):
    """
    Filters a list of values based on the standard deviation.
    Any compound which deviates more than max_sds standard deviations from the average is discarded.
    :param to_filter: A list of values to filter
    :param max_sds: The maximum amount of standard deviations a value can deviate from the average
    :return: a filtered list of values
    """
    mean = np.mean(to_filter)
    sd = np.std(to_filter)
    return [i for i in to_filter if mean - max_sds * sd < i < mean + max_sds * sd]


def min_max_scale(to_scale):
    """
    Performs min/max scaling on a list.
    :param to_scale: the list to scale
    :return: a scaled copy of the passed list
    """
    minimum = min(to_scale)
    maximum = max(to_scale)
    return [(i - minimum) / (maximum - minimum) for i in to_scale]


def sd_filter_boolean(to_filter, max_sds):
    """
    Filters a list of values based on the standard deviation.
    Any compound which deviates more than max_sds standard deviations from the average is discarded.
    :param to_filter: A list of values to filter
    :param max_sds: The maximum amount of standard deviations a value can deviate from the average
    :return: boolean, says if the value must be filtered
    """
    mean = np.mean(to_filter)
    sd = np.std(to_filter)
    return [mean - max_sds * sd < i < mean + max_sds * sd for i in to_filter]


if __name__ == '__main__':
    df = pd.read_csv("data/binding_data_final.tsv", sep="\t")
    filtered = df["IC50 (nm)"]
    min_max_scaled = min_max_scale(filtered)
    figure = plt.figure()

    figure.suptitle('Log10 of IC50', fontsize=16)
    figure = plt.hist(min_max_scaled, color='blue', edgecolor='black',
                      bins=100)
    plt.show()
