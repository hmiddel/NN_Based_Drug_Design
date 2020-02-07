import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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
    filtered = [i for i in log_scaled if mean - 3 * sd < i < mean + 3 * sd]
    minimum = min(filtered)
    maximum = max(filtered)
    min_max_scaled = [(i-minimum)/(maximum-minimum) for i in filtered]
    figure = plt.figure()
    figure.suptitle('Repartition of kinase families after embedding', fontsize=16)
    figure = plt.hist(min_max_scaled, color = 'blue', edgecolor = 'black',
         bins = 100)
    plt.show()
