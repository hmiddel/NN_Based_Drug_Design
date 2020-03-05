import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plots(x, y, label):
    x = np.log10(x)
    fig = plt.figure(figsize=(6, 12))

    ax1 = fig.add_subplot(311)
    ax1.set_title('real')
    ax1.hist(x)
    ax1.set_xlim([-2, 5])

    ax2 = fig.add_subplot(312)
    ax2.set_title('predicted')
    ax2.hist(y)
    ax2.set_xlim([-2, 5])

    ax3 = fig.add_subplot(313)
    ax3.scatter(x, y)
    ax3.set_xlim([-2, 5])
    ax3.set_ylim([-2, 5])
    plt.show()
    plt.savefig("data/model_visu_" + str(label) + ".png")


if __name__ == "__main__":
    data = pd.read_csv("data/predictions.tsv", sep="\t", usecols=["predicted IC50", "IC50 (nM)"],
                       dtype={"predicted IC50": np.float64, "IC50 (nM)": np.float64})
    plots(data["IC50 (nM)"], np.log10(data["predicted IC50"]), 'test')
