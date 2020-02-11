import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def sd_filter(to_filter, max_sds):
    return [i for i in to_filter if mean - max_sds * sd < i < mean + max_sds * sd]


def min_max_scale(to_scale):
    minimum = min(to_scale)
    maximum = max(to_scale)
    return [(i - minimum) / (maximum - minimum) for i in to_scale]


if __name__ == '__main__':
    df = pd.read_csv("data/binding_data.tsv", sep="\t")
    df = df[df["IC50 (nm)"] != 0]
    IC50s = df["IC50 (nm)"]
    print(IC50s[:5])
    log_scaled = np.log10(IC50s)
    print(sum([i > 20 for i in log_scaled]))
    mean = np.mean(log_scaled)
    sd = np.std(log_scaled)
    print(f"Mean: {mean}")
    print(f"Standard Deviation: {sd}")
    filtered = sd_filter(log_scaled, 3)

    min_max_scaled = min_max_scale(filtered)
    figure = plt.figure()

    figure.suptitle('Log10 of IC50', fontsize=16)
    figure = plt.hist(min_max_scaled, color='blue', edgecolor='black',
                      bins=100)
    plt.show()
