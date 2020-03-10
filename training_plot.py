import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random


def plots(x, y, label=None, save=False):
    fig = plt.figure(figsize=(5, 12))

    ax1 = fig.add_subplot(311)
    ax1.set_title('real')
    ax1.hist(x)
    ax1.set_xlim(min(min(y),min(x)),max(max(x),max(y)))

    ax2 = fig.add_subplot(312)
    ax2.set_title('predicted')
    ax2.hist(y)
    ax2.set_xlim(min(min(y),min(x)),max(max(x),max(y)))

    ax3 = fig.add_subplot(313)
    ax3.scatter(x, y)
    ax3.axis('equal')
    plt.show()
    if save:
        plt.savefig('data/' + str(label) + '.png')


if __name__ == "__main__":
    x = random.sample(range(10000), 100)
    y = random.sample(range(100), 100)
    plots(x, y)
