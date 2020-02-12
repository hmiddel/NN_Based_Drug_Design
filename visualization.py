import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def sd_filter(to_filter, max_sds):
    mean = np.mean(to_filter)
    sd = np.std(to_filter)
    return [i for i in to_filter if mean - max_sds * sd < i < mean + max_sds * sd]


def min_max_scale(to_scale):
    minimum = min(to_scale)
    maximum = max(to_scale)
    return [(i - minimum) / (maximum - minimum) for i in to_scale]


if __name__ == '__main__':
    df = pd.read_csv("data/binding_data_final.tsv", sep="\t")
    filtered = df["IC50 (nm)"]
    min_max_scaled = min_max_scale(filtered)
    figure = plt.figure()

    figure.suptitle('Log10 of IC50', fontsize=16)
    figure = plt.hist(min_max_scaled, color='blue', edgecolor='black',
                      bins=100)
    plt.show()
