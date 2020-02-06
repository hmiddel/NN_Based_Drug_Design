import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    df = pd.read_csv("data/binding_data.tsv", sep="\t")
    df = df[df["IC50 (nm)"] != 0]
    df = df[df["IC50 (nm)"] < 1000000]
    IC50s = df["IC50 (nm)"]
    print(IC50s[:5])
    scaled = np.log10(IC50s)
    print(sum([i > 20 for i in scaled]))
    figure = plt.figure()
    figure.suptitle('Repartition of kinase families after embedding', fontsize=16)
    figure = plt.hist(scaled, color = 'blue', edgecolor = 'black',
         bins = 100)
    plt.show()
